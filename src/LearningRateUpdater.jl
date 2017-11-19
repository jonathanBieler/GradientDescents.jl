
struct ConstLearningRate <: LearningRateUpdater
    η::Float64
end
ConstLearningRate() = ConstLearningRate(1e-3)


init(lr::ConstLearningRate) = lr.η
update(lr::ConstLearningRate,η) = zero(η)