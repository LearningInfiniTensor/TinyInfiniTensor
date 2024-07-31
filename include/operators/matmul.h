#pragma once
#include "core/operator.h"

namespace infini
{
    /**
     * @brief Matrix multiplication.
     *
     */
    class MatmulObj : public OperatorObj
    {
    private:
        // InfiniTensor assumes a row-major tensor layout. `transA`=false means
        // default dims, true means A should be transposed before matmul. This is in
        // oppsite to the column-major BLAS.
        bool transA, transB;

        // Auxiliary attributes which are not a part of operator attributes.
        int m, n, k;

    public:
        /**
         * @brief Matmul operator with batch broadcast and tensor transpose
         * supports. Only one tensor with singe batch can be broadcasted due to the
         * BLAS interface restriction. Tranpose indicates whether the last two
         * dimensions should be transposed before Matmul and does not affect other
         * leading dimensions.
         *
         * Matmul show how operators are defined in InfiniTensor. The constructor of
         * an operator can create output tensors for the operator or not, which
         * depends on `graph`.
         *
         * @param graph The computation graph that this operator belongs to.
         * @param A The input tensor.
         * @param B The input tensor.
         * @param C C is the output of Matmul. If outputs are going to be created in
         * the constructor, C should be an empty Ref.
         * @param transA If matrix A should be transposed when computing.
         * @param transB If matrix B should be transposed when computing.
         */
        MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C,
                  bool transA = false, bool transB = false);
        OP_CLONE(MatmulObj);

        std::string toString() const override;
        optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

        int numInputs() const override { return inputs.size(); }
        int numOutputs() const override { return 1; }

        bool getTransA() const { return transA; }
        bool getTransB() const { return transB; }
        void setTransA(bool transA) { this->transA = transA; }
        void setTransB(bool transB) { this->transB = transB; }
        int getM() const { return m; }
        int getN() const { return n; }
        int getK() const { return k; }
    };

} // namespace infini