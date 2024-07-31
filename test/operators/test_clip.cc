#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

    TEST(Clip, ShapeInference)
    {
        // Runtime
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 2, 2, 3}, DataType::Float32);
        float min = 1.0;
        float max = 4.0;
        auto op = g->addOp<ClipObj>(i0, nullptr, min, max);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 2, 2, 3}));
        EXPECT_EQ(op->getOutDType(), (DataType::Float32));
    }

} // namespace infini
