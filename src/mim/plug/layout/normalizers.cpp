#include <optional>
#include <string>
#include <utility>

#include <mim/axm.h>
#include <mim/def.h>
#include <mim/tuple.h>
#include <mim/union.h>
#include <mim/world.h>

#include "mim/plug/core/core.h"
#include "mim/plug/layout/layout.h"

namespace mim::plug::layout {

namespace {

/// A concrete `IntTuple` value unrolled for structural matching.
struct View {
    bool is_leaf;
    const Def* val; ///< `leaf`: the extent; `nest`: the child count `n`.
    DefVec kids;    ///< `nest` only.

    static View leaf(const Def* a) { return View{true, a, {}}; }
};

/// Matches a *constructor* value; a variable (dynamic shape) yields `std::nullopt`.
std::optional<View> view(const Def* t) {
    auto inj = t->isa<Inj>();
    if (!inj) return {};

    if (inj->index() == 0) return View::leaf(inj->value());

    auto pair = inj->value();
    if (!pair->isa<Tuple>()) return {};
    auto [n, xs] = pair->projs<2>();
    auto ln      = Lit::isa<u64>(n);
    if (!ln) return {};

    // `«0; T»` -> `[]` and `«1; T»` -> `T`, so the child list degenerates for `n < 2`.
    DefVec kids;
    if (*ln == 1)
        kids.push_back(xs);
    else if (*ln >= 2)
        for (u64 i = 0; i != *ln; ++i) kids.push_back(xs->proj(*ln, i));

    return View{false, n, std::move(kids)};
}

/// The product of all leaves (Lean's `size`); `nullptr` if a nested value is dynamic.
const Def* size_of(World& w, const View& v) {
    if (v.is_leaf) return v.val;

    const Def* res = w.lit_nat(1);
    for (auto k : v.kids) {
        auto kv = view(k);
        if (!kv) return nullptr;
        auto s = size_of(w, *kv);
        if (!s) return nullptr;
        res = w.call(core::nat::mul, Defs{res, s});
    }
    return res;
}

/// Same nesting, ignoring extents; `std::nullopt` if either side is dynamic.
std::optional<bool> congruent_of(const View& a, const View& b) {
    if (a.is_leaf != b.is_leaf) return false;
    if (a.is_leaf) return true;
    if (a.kids.size() != b.kids.size()) return false;

    for (size_t i = 0; i != a.kids.size(); ++i) {
        auto ka = view(a.kids[i]);
        auto kb = view(b.kids[i]);
        if (!ka || !kb) return {};
        auto c = congruent_of(*ka, *kb);
        if (!c) return {};
        if (!*c) return false;
    }
    return true;
}

/// The bounds predicate as a `Bool` Def, folded on literals and residualized otherwise.
const Def* build_contains(World& w, const View& shape, const View& coord) {
    if (shape.is_leaf != coord.is_leaf) return w.lit_ff();
    if (shape.is_leaf) return w.call(core::ncmp::l, Defs{coord.val, shape.val});
    if (shape.kids.size() != coord.kids.size()) return w.lit_ff();

    const Def* res = w.lit_tt();
    for (size_t i = 0; i != shape.kids.size(); ++i) {
        auto ks = view(shape.kids[i]);
        auto kc = view(coord.kids[i]);
        if (!ks || !kc) return nullptr;
        auto c = build_contains(w, *ks, *kc);
        if (!c) return nullptr;
        res = w.call(core::bit2::and_, w.lit_nat(0), Defs{res, c});
    }
    return res;
}

/// Like `build_contains`, but a `leaf` coord may be a flat index into a nested shape.
const Def* build_compatible(World& w, const View& shape, const View& coord) {
    if (shape.is_leaf && coord.is_leaf) return w.call(core::ncmp::l, Defs{coord.val, shape.val});
    if (!shape.is_leaf && coord.is_leaf) {
        auto s = size_of(w, shape);
        if (!s) return nullptr;
        return w.call(core::ncmp::l, Defs{coord.val, s});
    }
    if (shape.is_leaf && !coord.is_leaf) return w.lit_ff();
    if (shape.kids.size() != coord.kids.size()) return w.lit_ff();

    const Def* res = w.lit_tt();
    for (size_t i = 0; i != shape.kids.size(); ++i) {
        auto ks = view(shape.kids[i]);
        auto kc = view(coord.kids[i]);
        if (!ks || !kc) return nullptr;
        auto c = build_compatible(w, *ks, *kc);
        if (!c) return nullptr;
        res = w.call(core::bit2::and_, w.lit_nat(0), Defs{res, c});
    }
    return res;
}

/// `idx = idx₀ + size₀·(idx₁ + size₁·(…))` — the first axis is the most minor.
const Def* build_coord_to_idx(World& w, const View& shape, const View& coord) {
    if (shape.is_leaf != coord.is_leaf) return nullptr;
    if (shape.is_leaf) return coord.val;
    if (shape.kids.size() != coord.kids.size()) return nullptr;

    const Def* res = w.lit_nat(0);
    for (size_t i = shape.kids.size(); i-- != 0;) {
        auto ks = view(shape.kids[i]);
        auto kc = view(coord.kids[i]);
        if (!ks || !kc) return nullptr;
        auto idx = build_coord_to_idx(w, *ks, *kc);
        if (!idx) return nullptr;
        auto sz = size_of(w, *ks);
        if (!sz) return nullptr;
        res = w.call(core::nat::add, Defs{idx, w.call(core::nat::mul, Defs{sz, res})});
    }
    return res;
}

/// `nest (n, kids)`; the child list degenerates for 0/1 exactly as `«n; T»` does.
const Def* nest(World& w, const Def* it, const DefVec& kids) {
    return w.inj(it, 1, w.tuple({w.lit_nat(kids.size()), w.tuple(kids)}));
}

/// Pushes a (possibly already nested) index into `shape`; mirrors `Tuple.idx_to_coord`.
const Def* build_idx_to_coord(World& w, const Def* it, const View& shape, const View& idx) {
    if (shape.is_leaf) {
        if (!idx.is_leaf) return nullptr; // a `nest` index has no place in a `leaf` shape
        return w.inj(it, 0, idx.val);
    }

    if (idx.is_leaf) { // flat index: delinearize, first axis most minor
        DefVec kids;
        const Def* prefix = w.lit_nat(1);
        for (auto k : shape.kids) {
            auto ks = view(k);
            if (!ks) return nullptr;
            auto sz = size_of(w, *ks);
            if (!sz) return nullptr;
            auto q = w.call(core::nat::div, Defs{idx.val, prefix});
            auto r = w.call(core::nat::rem, Defs{q, sz});
            auto c = build_idx_to_coord(w, it, *ks, View::leaf(r));
            if (!c) return nullptr;
            kids.push_back(c);
            prefix = w.call(core::nat::mul, Defs{prefix, sz});
        }
        return nest(w, it, kids);
    }

    if (shape.kids.size() != idx.kids.size()) return nullptr; // pointwise
    DefVec kids;
    for (size_t i = 0; i != shape.kids.size(); ++i) {
        auto ks = view(shape.kids[i]);
        auto ki = view(idx.kids[i]);
        if (!ks || !ki) return nullptr;
        auto c = build_idx_to_coord(w, it, *ks, *ki);
        if (!c) return nullptr;
        kids.push_back(c);
    }
    return nest(w, it, kids);
}

} // namespace

const Def* normalize_rank(const Def* type, const Def*, const Def* arg) {
    auto& w = type->world();
    auto v  = view(arg);
    if (!v) return {};
    return v->is_leaf ? w.lit_nat(1) : v->val;
}

const Def* normalize_size(const Def* type, const Def*, const Def* arg) {
    auto& w = type->world();
    auto v  = view(arg);
    if (!v) return {};
    return size_of(w, *v);
}

const Def* normalize_congruent(const Def* type, const Def*, const Def* arg) {
    auto& w     = type->world();
    auto [a, b] = arg->projs<2>();
    auto va     = view(a);
    auto vb     = view(b);
    if (!va || !vb) return {};
    auto res = congruent_of(*va, *vb);
    if (!res) return {};
    return w.lit_bool(*res);
}

const Def* normalize_contains(const Def* type, const Def*, const Def* arg) {
    auto& w     = type->world();
    auto [a, b] = arg->projs<2>();
    auto va     = view(a);
    auto vb     = view(b);
    if (!va || !vb) return {};
    return build_contains(w, *va, *vb);
}

const Def* normalize_compatible(const Def* type, const Def*, const Def* arg) {
    auto& w     = type->world();
    auto [a, b] = arg->projs<2>();
    auto va     = view(a);
    auto vb     = view(b);
    if (!va || !vb) return {};
    return build_compatible(w, *va, *vb);
}

const Def* normalize_coord_to_idx(const Def* type, const Def*, const Def* arg) {
    auto& w          = type->world();
    auto [shape, co] = arg->projs<2>();
    auto vs          = view(shape);
    auto vc          = view(co);
    if (!vs || !vc) return {};
    return build_coord_to_idx(w, *vs, *vc);
}

const Def* normalize_idx_to_coord(const Def* type, const Def*, const Def* arg) {
    auto& w         = type->world();
    auto [shape, x] = arg->projs<2>();
    auto vs         = view(shape);
    if (!vs) return {};
    return build_idx_to_coord(w, shape->type(), *vs, View::leaf(x));
}

/// `at (shape, x)` with `x: Idx (size shape)`; the bound is in the type, so no check is needed.
const Def* normalize_at(const Def* type, const Def*, const Def* arg) {
    auto& w         = type->world();
    auto [shape, x] = arg->projs<2>();
    auto vs         = view(shape);
    if (!vs) return {};
    auto nat = w.call<core::bitcast>(w.type_nat(), x);
    return build_idx_to_coord(w, shape->type(), *vs, View::leaf(nat));
}

/// `assert (cond, val, msg)`: `val` on `tt`, a hard compile-time error on `ff`, stuck otherwise.
const Def* normalize_assert(const Def* type, const Def*, const Def* arg) {
    auto& w               = type->world();
    auto [cond, val, msg] = arg->projs<3>();

    if (cond == w.lit_tt()) return val;
    if (cond == w.lit_ff()) {
        auto s = tuple2str(msg);
        if (s.empty()) s = "layout constraint failed";
        val->blame("{}", s).bail();
    }
    return {};
}

/// `bounded (shape, x)`: the singleton proof `«compatible (shape, leaf x)»`.
const Def* normalize_bounded(const Def* type, const Def*, const Def* arg) {
    auto& w         = type->world();
    auto [shape, x] = arg->projs<2>();
    auto vs         = view(shape);
    if (!vs) return {};

    auto pred = build_compatible(w, *vs, View::leaf(x));
    if (!pred) return {};
    if (pred == w.lit_ff()) x->blame("index out of bounds for shape").bail();
    if (pred == w.lit_tt()) return w.wrap(w.lit_tt());
    return {};
}

MIM_layout_NORMALIZER_IMPL

} // namespace mim::plug::layout
