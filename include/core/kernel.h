#pragma once
#include "core/common.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "utils/operator_utils.h"
#include <functional>

namespace infini
{

    class RuntimeObj;

    class Kernel
    {
    public:
        Kernel() {}
        virtual ~Kernel() {}

        /**
         * @brief Executes an op with a default parameter.
         */
        virtual void compute(const Operator &op,
                             const RuntimeObj *context) const = 0;
    };

    class KernelRegistry
    {
    public:
        using KernelRecord =
            tuple<Kernel *const, const string, const int>; // Kernel, name, ID

    private:
        std::map<KernelAttrs, KernelRecord> kernels;
        int nKernels = 0;

    public:
        ~KernelRegistry()
        {
            for (auto &[k, v] : kernels)
                delete std::get<0>(v);
        }
        static KernelRegistry &getInstance()
        {
            static KernelRegistry instance;
            return instance;
        }
        bool registerKernel(const KernelAttrs &key, Kernel *kernel, string name)
        {
            IT_ASSERT(kernels.find(key) == kernels.end(),
                      "Kernel already registered");
            kernels.emplace(key, KernelRecord{kernel, name, ++nKernels});
            return true;
        }
        Kernel *getKernel(const KernelAttrs &kernelAttrs) const
        {
            auto it = kernels.find(kernelAttrs);
            IT_ASSERT(it != kernels.end(), "Kernel not found for key {" +
                                               get_kernel_attrs_str(kernelAttrs) +
                                               "}");
            return std::get<0>(it->second);
        }
        const KernelRecord &getKernelItem(const KernelAttrs &kernelAttrs) const
        {
            return kernels.at(kernelAttrs);
        }
    };

    class CpuKernelWithoutConfig : public Kernel
    {
    public:
        virtual void compute(const Operator &op,
                             const RuntimeObj *context) const = 0;
    };

} // namespace infini

#define _REGISTER_KERNEL_1(device, opType, kernel, name, cnt)                 \
    namespace infini                                                          \
    {                                                                         \
        static const bool _CAT(_register_kernel_, cnt) =                      \
            KernelRegistry::getInstance().registerKernel(KernelAttrs{device,  \
                                                                     opType}, \
                                                         new kernel(), name); \
    }

#define REGISTER_KERNEL(device, opType, kernel, name) \
    _REGISTER_KERNEL_1(device, opType, kernel, name, __COUNTER__)
