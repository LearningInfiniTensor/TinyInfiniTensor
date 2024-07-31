#include "core/op_type.h"

namespace infini
{
    const char *OpType::toString() const
    {
#define CASE(NAME)     \
    case OpType::NAME: \
        return #NAME

        switch (type)
        {
            CASE(Unknown);
            CASE(Add);
            CASE(Sub);
            CASE(Mul);
            CASE(Div);
            CASE(Cast);
            CASE(Clip);
            CASE(Relu);
            CASE(Transpose);
            CASE(Concat);
            CASE(MatMul);

        default:
            return "Unknown";
        }

#undef CASE
    }

} // namespace infini
