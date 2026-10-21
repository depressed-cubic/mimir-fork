#include <optional>
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

/// `Coord shape` as a type: `Idx` at a leaf, a product of `Coord` at a nest.
const Def* coord_ty(World& w, const View& v) {
    if (v.is_leaf) return w.type_idx(v.val);

    DefVec tys;
    tys.reserve(v.kids.size());
    for (auto k : v.kids) tys.push_back(w.app(w.annex<Coord>(), k));
    return w.sigma(tys);
}

/// Pushes a *Nat* index `xn` into `Coord shape`.
const Def* build_coord(World& w, const View& v, const Def* xn) {
    if (v.is_leaf) return w.call<core::idx>(w.lit_nat(0), v.val, xn);

    DefVec kids;
    const Def* prefix = w.lit_nat(1);
    for (auto k : v.kids) {
        auto kv = view(k);
        if (!kv) return nullptr;
        auto sz = size_of(w, *kv);
        if (!sz) return nullptr;
        auto q = w.call(core::nat::div, Defs{xn, prefix});
        auto r = w.call(core::nat::rem, Defs{q, sz});
        auto c = build_coord(w, *kv, r);
        if (!c) return nullptr;
        kids.push_back(c);
        prefix = w.call(core::nat::mul, Defs{prefix, sz});
    }
    return w.tuple(kids);
}

/// Pulls a `Coord shape` out to a *Nat* index (first axis most minor).
const Def* coord_to_idx_nat(World& w, const View& v, const Def* c) {
    if (v.is_leaf) return w.call<core::bitcast>(w.type_nat(), c);

    const Def* res = w.lit_nat(0);
    for (size_t i = v.kids.size(); i-- != 0;) {
        auto kv = view(v.kids[i]);
        if (!kv) return nullptr;
        auto idx = coord_to_idx_nat(w, *kv, c->proj(v.kids.size(), i));
        if (!idx) return nullptr;
        auto sz = size_of(w, *kv);
        if (!sz) return nullptr;
        res = w.call(core::nat::add, Defs{idx, w.call(core::nat::mul, Defs{sz, res})});
    }
    return res;
}

/// Forgets the bounds of `c: Coord shape`, yielding an `IntTuple`.
const Def* erase_rec(World& w, const Def* it, const View& v, const Def* c) {
    if (v.is_leaf) return w.inj(it, 0, w.call<core::bitcast>(w.type_nat(), c));

    DefVec kids;
    for (size_t i = 0; i != v.kids.size(); ++i) {
        auto kv = view(v.kids[i]);
        if (!kv) return nullptr;
        auto e = erase_rec(w, it, *kv, c->proj(v.kids.size(), i));
        if (!e) return nullptr;
        kids.push_back(e);
    }
    return w.inj(it, 1, w.tuple({w.lit_nat(kids.size()), w.tuple(kids)}));
}

/// `Congruent a b` as a type: `[]` (unit) when congruent, `⊥` when not.
const Def* congruent_ty(World& w, const View& a, const View& b) {
    if (a.is_leaf != b.is_leaf) return w.type_bot();
    if (a.is_leaf) return w.sigma();
    if (a.kids.size() != b.kids.size()) return w.type_bot();

    DefVec tys;
    for (size_t i = 0; i != a.kids.size(); ++i) {
        auto ka = view(a.kids[i]);
        auto kb = view(b.kids[i]);
        if (!ka || !kb) return nullptr;
        auto t = congruent_ty(w, *ka, *kb);
        if (!t) return nullptr;
        if (t == w.type_bot()) return t;
        tys.push_back(t);
    }
    return w.sigma(tys);
}

/// The canonical inhabitant of `Congruent a b`; `nullptr` for a non-congruent/`⊥` side.
const Def* mk_congruent_rec(World& w, const View& a, const View& b) {
    if (a.is_leaf != b.is_leaf) return nullptr;
    if (a.is_leaf) return w.tuple();
    if (a.kids.size() != b.kids.size()) return nullptr;

    DefVec pfs;
    for (size_t i = 0; i != a.kids.size(); ++i) {
        auto ka = view(a.kids[i]);
        auto kb = view(b.kids[i]);
        if (!ka || !kb) return nullptr;
        auto p = mk_congruent_rec(w, *ka, *kb);
        if (!p) return nullptr;
        pfs.push_back(p);
    }
    return w.tuple(pfs);
}

/// The nested inner product `Σ cᵢ · strideᵢ`.
const Def* crd2idx_rec(World& w, const View& sh, const View& st, const Def* c) {
    if (sh.is_leaf) {
        auto cv = w.call<core::bitcast>(w.type_nat(), c);
        return w.call(core::nat::mul, Defs{cv, st.val});
    }

    const Def* res = w.lit_nat(0);
    for (size_t i = 0; i != sh.kids.size(); ++i) {
        auto ksh = view(sh.kids[i]);
        auto kst = view(st.kids[i]);
        if (!ksh || !kst) return nullptr;
        auto t = crd2idx_rec(w, *ksh, *kst, c->proj(sh.kids.size(), i));
        if (!t) return nullptr;
        res = w.call(core::nat::add, Defs{res, t});
    }
    return res;
}

} // namespace

const Def* normalize_coord_ty(const Def* type, const Def*, const Def* arg) {
    auto& w = type->world();
    auto v  = view(arg);
    if (!v) return {};
    return coord_ty(w, *v);
}

const Def* normalize_size(const Def* type, const Def*, const Def* arg) {
    auto& w = type->world();
    auto v  = view(arg);
    if (!v) return {};
    return size_of(w, *v);
}

const Def* normalize_rank(const Def* type, const Def*, const Def* arg) {
    auto& w = type->world();
    auto v  = view(arg);
    if (!v) return {};
    return v->is_leaf ? w.lit_nat(1) : v->val;
}

const Def* normalize_idx_to_coord(const Def* type, const Def*, const Def* arg) {
    auto& w         = type->world();
    auto [shape, x] = arg->projs<2>();
    auto v          = view(shape);
    if (!v) return {};
    auto xn = w.call<core::bitcast>(w.type_nat(), x);
    return build_coord(w, *v, xn);
}

const Def* normalize_coord_to_idx(const Def* type, const Def*, const Def* arg) {
    auto& w          = type->world();
    auto [shape, c]  = arg->projs<2>();
    auto v           = view(shape);
    if (!v) return {};
    auto nat = coord_to_idx_nat(w, *v, c);
    if (!nat) return {};
    auto sz = size_of(w, *v);
    if (!sz) return {};
    return w.call<core::idx>(w.lit_nat(0), sz, nat);
}

const Def* normalize_erase(const Def* type, const Def*, const Def* arg) {
    auto& w         = type->world();
    auto [shape, c] = arg->projs<2>();
    auto v          = view(shape);
    if (!v) return {};
    return erase_rec(w, shape->type(), *v, c);
}

const Def* normalize_congruent_ty(const Def* type, const Def*, const Def* arg) {
    auto& w     = type->world();
    auto [a, b] = arg->projs<2>();
    auto va     = view(a);
    auto vb     = view(b);
    if (!va || !vb) return {};
    return congruent_ty(w, *va, *vb);
}

const Def* normalize_mk_congruent(const Def* type, const Def*, const Def* arg) {
    auto& w     = type->world();
    auto [a, b] = arg->projs<2>();
    auto va     = view(a);
    auto vb     = view(b);
    if (!va || !vb) return {};

    auto ty = congruent_ty(w, *va, *vb);
    if (!ty) return {};
    if (ty == w.type_bot()) a->blame("shape and stride are not congruent").bail();
    return mk_congruent_rec(w, *va, *vb);
}

const Def* normalize_crd2idx(const Def* type, const Def*, const Def* arg) {
    auto& w         = type->world();
    auto [l, c]     = arg->projs<2>();
    auto vsh        = view(l->proj(3, 0));
    auto vst        = view(l->proj(3, 1));
    if (!vsh || !vst) return {};
    return crd2idx_rec(w, *vsh, *vst, c);
}

MIM_layout_NORMALIZER_IMPL

} // namespace mim::plug::layout
