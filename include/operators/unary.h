#pragma once
#include "core/operator.h"

namespace infini
{
  /**
   * @brief The base class for unary operators.
   *
   */
  class UnaryObj : public OperatorObj
  {
  public:
    /**
     * @brief Construct a new Unary object.
     *
     * @param type Operator type.
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor.
     */
    UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
  };

  class ClipObj : public OperatorObj
  {
  public:
    ClipObj(GraphObj *graph, Tensor input, Tensor output,
            std::optional<float> min, std::optional<float> max);
    OP_CLONE(ClipObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    std::optional<float> getMin() const { return minValue; };
    std::optional<float> getMax() const { return maxValue; };
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    std::optional<float> minValue, maxValue;
  };

  enum class CastType
  {
    Float2Float16 = 0,
    Float2Int64,
    Float2Int32,
    Float2Int16,
    Float2Int8,
    Float2BFloat16,
    Int322Float,
    Int322Int8,
    Int322Int16,
    Int322Int64,
    Int162Float,
    Int162Int32,
    Int82Float,
    Int82Int16,
    Int82Int32,
    Uint82Float,
    Uint82Int32,
    Uint82Int64,
    Int642Int32,
    Int642Uint32,
    Int642Float,
    Uint322Int64,
    Float162Float,
    BFloat162Float,
    Float2Float,
  };

  class CastObj : public OperatorObj
  {
  public:
    CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type);
    OP_CLONE(CastObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

    std::string toString() const override;
    CastType getType() const { return castType; }
    DataType getOutputDataType() const;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    CastType castType;
  };

#define DEFINE_UNARY_OBJ(prefix, type)                        \
  class prefix##Obj : public UnaryObj                         \
  {                                                           \
  public:                                                     \
    prefix##Obj(GraphObj *graph, Tensor input, Tensor output) \
        : UnaryObj(type, graph, input, output) {}             \
    OP_CLONE(prefix##Obj);                                    \
  };

  DEFINE_UNARY_OBJ(Relu, OpType::Relu)
}; // namespace infini
