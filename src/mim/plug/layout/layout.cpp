#include "mim/plug/layout/layout.h"

#include <mim/phase.h>
#include <mim/plugin.h>

using namespace mim;

/// Registers normalizers as well as Phase%s and Pass%es for the Axm%s of this Plugin.
MIM_PLUGIN_ENTRY(layout) { plugin.register_normalizers = plug::layout::register_normalizers; }
