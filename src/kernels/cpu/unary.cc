#include "operators/unary.h"
#include "core/kernel.h"

namespace infini
{
    class NativeUnary : public CpuKernelWithoutConfig
    {
        template <typename T>
        static T reluCompute(T val)
        {
            return std::max(T(0), val);
        }

        template <typename T>
        void doCompute(const Operator &_op, const RuntimeObj *context) const
        {
            auto op = as<UnaryObj>(_op);
            T *inptr = op->getInputs(0)->getRawDataPtr<T *>();
            T *outptr = op->getOutput()->getRawDataPtr<T *>();

            auto outDim = op->getOutput()->getDims();
            auto n = op->getOutput()->size();

            T (*_doCompute)
            (T val);
            switch (op->getOpType().underlying())
            {
            case OpType::Relu:
                _doCompute = reluCompute<T>;
                break;
            default:
                IT_TODO_HALT();
            }

            for (size_t offset = 0; offset < n; offset++)
            {
                outptr[offset] = _doCompute(inptr[offset]);
            }
        }

        void compute(const Operator &_op,
                     const RuntimeObj *context) const override
        {
#define CASE(N) \
    case N:     \
        doCompute<DT<N>::t>(_op, context)

            int dataTypeIdx = _op->getDType().getIndex();
            switch (dataTypeIdx)
            {
                CASE(1); // DataType::Float32
                break;
                CASE(12); // DataType::UInt32
                break;
            default:
                IT_TODO_HALT();
            }
        }
    };

    class Clip : public CpuKernelWithoutConfig
    {
        template <typename T>
        void doCompute(const Operator &_op, const RuntimeObj *context) const
        {
            auto op = as<ClipObj>(_op);
            T *inptr = op->getInputs(0)->getRawDataPtr<T *>();
            T *outptr = op->getOutput()->getRawDataPtr<T *>();
            auto minValue = op->getMin();
            auto maxValue = op->getMax();

            auto n = op->getOutput()->size();
            for (size_t offset = 0; offset < n; offset++)
            {
                auto val = *inptr++;
                *outptr++ = (minValue && val < *minValue)   ? *minValue
                            : (maxValue && val > *maxValue) ? *maxValue
                                                            : val;
            }
        }

        void compute(const Operator &_op,
                     const RuntimeObj *context) const override
        {
#define CASE(N) \
    case N:     \
        doCompute<DT<N>::t>(_op, context)

            int dataTypeIdx = _op->getDType().getIndex();
            switch (dataTypeIdx)
            {
                CASE(1); // DataType::Float32
                break;
                CASE(12); // DataType::UInt32
                break;
            default:
                IT_TODO_HALT();
            }
        }
    };

    REGISTER_KERNEL(Device::CPU, OpType::Relu, NativeUnary, "reluNaive_CPU");
    REGISTER_KERNEL(Device::CPU, OpType::Clip, Clip, "Clip_CPU");

}; // namespace infini
