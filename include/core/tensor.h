#pragma once
#include "core/blob.h"
#include "core/data_type.h"
#include "core/object.h"
#include "core/runtime.h"
#include <cmath>
#include <cstring>
#include <fstream>

namespace infini
{
    class GraphObj;
    using ShapeElem = int;
    using Shape = vector<ShapeElem>;
    class TensorObj : public Object
    {
        friend class GraphObj;

    protected:
        int dim;

        DataType dtype;
        vector<WRef<OperatorObj>> targets;
        WRef<OperatorObj> source;
        Blob data;
        Runtime runtime;

    private:
        Shape shape;
        size_t _size; // Cache of Π(shape).
        Fuid fuid;    // Cloned tensors share the same id. Tensors constructed from
                      // scratch have a new id.

    public:
        TensorObj(Shape shape, DataType dtype, Runtime runtime);
        virtual ~TensorObj() {}
        string toString() const override;

        size_t size() const { return _size; }
        size_t getBytes() const { return _size * dtype.getSize(); }

        Shape getDims() const { return shape; }
        void setShape(Shape shape_);
        size_t getRank() const { return shape.size(); }
        UidBaseType getFuid() const { return fuid; }

        void setData(
            std::function<void(void *, size_t, DataType)> const &generator) const;

        void setDataBlob(const Blob &blob);

        void printData() const;
        bool equalData(const Tensor &rhs, double relativeError = 1e-6) const;

        template <typename T>
        bool equalData(const vector<T> &dataVector)
        {
            IT_ASSERT(size() == dataVector.size());
            IT_ASSERT(DataType::get<T>() == dtype.cpuTypeInt());
            return equalDataImpl(getRawDataPtr<T *>(), dataVector.data(), size());
        }

        template <typename T>
        T getRawDataPtr() const
        {
            static_assert(std::is_pointer_v<T>,
                          "Raw data pointer has a type of pointer");
            IT_ASSERT(data != nullptr);
            return data->getPtr<T>();
        }

        DataType getDType() const { return dtype; }
        Runtime getRuntime() const { return runtime; }

        OpVec getTargets() const { return wrefs_to_refs(targets); }
        Operator getSource() const { return source.lock(); }

    private:
        template <class T>
        string dataToString() const
        {
            std::stringstream builder;
            builder << "Tensor: " << guid << std::endl;

            auto numDims = shape.size();
            auto dimSzVec = vector<int>(numDims, 1);
            auto ptr = data->getPtr<T *>();
            dimSzVec[numDims - 1] = shape[numDims - 1];

            for (int i = numDims - 1; i != 0; --i)
                dimSzVec[i - 1] = dimSzVec[i] * shape[i - 1];

            for (size_t i = 0, iEnd = size(); i < iEnd; ++i)
            {
                for (size_t j = 0; j < numDims; ++j)
                    if (i % dimSzVec[j] == 0)
                        builder << "[";

                builder << ptr[i];
                for (size_t j = 0; j < numDims; ++j)
                    if ((int)i % dimSzVec[j] == dimSzVec[j] - 1)
                        builder << "]";

                if (i != size() - 1)
                    builder << ", ";

                auto column = (size_t)dimSzVec[numDims - 1];
                if (i % column == column - 1)
                    builder << std::endl;
            }
            return builder.str();
        }

        template <typename T>
        bool equalDataImpl(const T *a, const T *b, size_t size,
                           double relativeError = 1e-6) const
        {
            for (size_t i = 0; i < size; ++i)
            {
                if constexpr (std::is_integral_v<T>)
                {
                    if (a[i] != b[i])
                        return false;
                }
                else if constexpr (std::is_floating_point_v<T>)
                {
                    if (std::min(fabs(a[i]), fabs(b[i])) == 0. &&
                        fabs(a[i] - b[i]) > relativeError)
                    {
                        printf("Error on %lu: %f %f\n", i, a[i], b[i]);
                        return false;
                    }
                    else if (std::min(fabs(a[i]), fabs(b[i])) != 0. &&
                             fabs(a[i] - b[i]) /
                                     std::max(fabs(a[i]), fabs(b[i])) >
                                 relativeError)
                    {
                        printf("Error on %lu: %f %f\n", i, a[i], b[i]);
                        return false;
                    }
                }
                else
                {
                    static_assert(!sizeof(T), "Unsupported data type");
                }
            }
            return true;
        }

        void addTarget(const Operator &op) { targets.emplace_back(op); }
        void setSource(const Operator &op) { source = op; }
        void removeTarget(const Operator &op)
        {
            for (auto itr = targets.begin(); itr != targets.end();)
            {
                if (itr->lock() == op)
                    itr = targets.erase(itr);
                else
                    ++itr;
            }
        }
    };

} // namespace infini
