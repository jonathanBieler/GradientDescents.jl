cd("/Users/jbieler/.julia/v0.6/GradientDescents/src/")
include("/Users/jbieler/.julia/v0.6/GradientDescents/src/GradientDescents.jl")

G = GradientDescents

## Tests

using BlackBoxOptimizationBenchmarking
const BBOB = BlackBoxOptimizationBenchmarking
include("plots.jl")

type Opt
    niter
end

G = GradientDescents
F = BBOB.F1
f = p -> 1e1*F.f(p)

## simplest one

gd = G.GradientDescent(
    G.StaticGradient(f),
    G.DirectDescent(G.SimpleMomentum(3e-200)),
    G.ConstLearningRate(),
    G.SimpleTerminator(),
    G.FullTracer(),
    (p,f) -> nothing#println(f(p))
)

@show G.optimize(gd, ones(2), Opt(10_000)).p ≈ F.x_opt[1:2]

      
plot_trace(gd.tracer)

##

mfit = Optim.optimize(f,ones(2),Optim.GradientDescent())
mfit.minimizer ≈ F.x_opt[1:2]

## 

using  LineSearches
linesearch = HagerZhang()
linesearch(1)

## AdaGrad with HypergradientDescent

F = BBOB.F2
f = p -> 1e1*BBOB.F2.f(p)

#

gd = G.GradientDescent(
    G.StaticGradient(f),
    G.AdaGrad(G.SimpleMomentum(1e-10)),
#    G.HypergradientDescent(1e-12,1e-15),s
    G.HypergradientDescent(),
    G.SimpleTerminator(),
    G.FullTracer(),
    (p,f) -> nothing
)

G.optimize(gd, 3*randn(2), Opt(10_000)).p ≈ F.x_opt[1:2]
#G.optimize(gd, [-3; 5], Opt(10_000)).p ≈ F.x_opt[1:2]

plot_trace(gd.tracer)

##

mfit = Optim.optimize(f,ones(2),Optim.GradientDescent())
mfit.minimizer ≈ F.x_opt[1:2]

##

plot_fun(F,gd.tracer)

## SGD

include("/Users/jbieler/.julia/v0.6/GradientDescents/src/GradientDescents.jl")
G = GradientDescents

model(x,p) = p[1] + p[2]*sin(x) + p[3]*x

x = 10*rand(1000)
y = model(x,[3 3 3]) + randn(size(x))*0.1
data = collect(zip(x,y))

f = (p,data) -> sum(abs2(model(data[i][1],p) - data[i][2]) for i=1:length(data))

batch_size = 200

gd = G.GradientDescent(
    G.MiniBatch(f,data,batch_size),
    G.AdaGrad(G.SimpleMomentum(1e-5)),
    G.HypergradientDescent(),
    G.SimpleTerminator(),
    G.FullTracer(),
    (p,f) -> nothing
)

G.optimize(gd, 3*randn(3), Opt(10_000)).p

plot_trace(gd.tracer)

##