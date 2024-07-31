#include "core/tensor.h"
#include "core/blob.h"
#include "core/operator.h"
#include "core/runtime.h"
#include <cstring>
#include <numeric>

namespace infini {

    TensorObj::TensorObj(Shape shape_, DataType dtype, Runtime runtime)
        : dim(shape_.size()), dtype(dtype), runtime(runtime), shape(std::move(shape_)),
          _size(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies{})) {}

    string TensorObj::toString() const
    {
        // Convert data pointer to string
        std::stringstream ss;
        if (data != nullptr)
            ss << data->getPtr<void *>();
        else
            ss << "nullptr data";
        string ret = "Tensor " + std::to_string(guid) + ", Fuid " +
                     std::to_string(fuid) + ", shape " + vecToString(shape) +
                     ", dtype " + dtype.toString() + ", " + runtime->toString() +
                     ", " + ss.str() + "\n";
        vector<UidBaseType> targetGuids;
        for (const auto &op : targets)
            targetGuids.emplace_back(op.lock()->getGuid());
        if (auto o = source.lock())
            ret += ", source " + std::to_string(o->getGuid());
        else
            ret += ", source None";
        ret += ", targets " + vecToString(targetGuids);
        return ret;
    }

void TensorObj::setShape(Shape shape_) {
    shape = shape_;
    size_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                  [](auto acc, auto x) { return acc * x; });
    _size = size;
}

void TensorObj::printData() const {
    IT_ASSERT(data != nullptr);
    if (!runtime->isCpu())
        IT_TODO_HALT();

#define TRY_PRINT(N)                                                           \
    if (dtype == DataType(N))                                                  \
        std::cout << dataToString<DT<N>::t>() << std::endl;

    TRY_PRINT(0)           // fmt: new line
    else TRY_PRINT(1)      //
        else TRY_PRINT(2)  //
        else TRY_PRINT(3)  //
        else TRY_PRINT(4)  //
        else TRY_PRINT(5)  //
        else TRY_PRINT(6)  //
        else TRY_PRINT(7)  //
        else TRY_PRINT(8)  //
        else TRY_PRINT(9)  //
        else TRY_PRINT(10) //
        else TRY_PRINT(11) //
        else TRY_PRINT(12) //
        else TRY_PRINT(13) //
        else TRY_PRINT(16) //
        else IT_TODO_HALT();

#undef TRY_PRINT
}

bool TensorObj::equalData(const Tensor &rhs, double relativeError) const {
    IT_ASSERT(data != nullptr);
    IT_ASSERT(rhs->data != nullptr);
    IT_ASSERT(getDType() == rhs->getDType());
    IT_ASSERT(runtime->isCpu());
    IT_ASSERT(rhs->getRuntime()->isCpu());
    if (size() != rhs->size())
        return false;

#define TEST_EQUAL(N)                                                          \
    if (dtype == DataType(N))                                                  \
        return equalDataImpl(getRawDataPtr<DT<N>::t *>(),                      \
                             rhs->getRawDataPtr<DT<N>::t *>(), size(),         \
                             relativeError);

    TEST_EQUAL(0)           // fmt: new line
    else TEST_EQUAL(1)      //
        else TEST_EQUAL(2)  //
        else TEST_EQUAL(3)  //
        else TEST_EQUAL(4)  //
        else TEST_EQUAL(5)  //
        else TEST_EQUAL(6)  //
        else TEST_EQUAL(7)  //
        else TEST_EQUAL(8)  //
        else TEST_EQUAL(9)  //
        else TEST_EQUAL(10) //
        else TEST_EQUAL(11) //
        else TEST_EQUAL(12) //
        else TEST_EQUAL(13) //
        else TEST_EQUAL(16) //
        else IT_TODO_HALT();

#undef TEST_EQUAL
}

void TensorObj::setData(
    const std::function<void(void *, size_t, DataType)> &generator) const {
    IT_ASSERT(data != nullptr);
    generator(getRawDataPtr<void *>(), size(), dtype);
}

void TensorObj::setDataBlob(const Blob &blob) { this->data = blob; }

}; // namespace infini
