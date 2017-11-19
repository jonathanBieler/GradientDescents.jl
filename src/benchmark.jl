include("/Users/bieler/.julia/v0.6/GradientDescents/src/GradientDescents.jl")

## Tests

using BBOBFunctions

type Opt
    niter
end

G = GradientDescents
F = BBOBFunctions.F1
f = F.f


##

gd = G.GradientDescent(
    G.StaticGradient(f),
    G.DirectDescent(),
    G.ConstLearningRate(1e-2),
    G.SimpleTerminator(),
    G.FullTracer(),
    (p,f) -> nothing#println(f(p))
)

@show G.optimize(gd, rand(3), Opt(1000)).p ≈ BBOBFunctions.F1.x_opt[1:3]

##

gd = G.GradientDescent(
    G.StaticGradient(f),
    G.AdaGrad(),
    G.ConstLearningRate(0.5),
    G.SimpleTerminator(),
    G.FullTracer(),
    (p,f) -> nothing
)

G.optimize(gd, 3*randn(2), Opt(2000)).p ≈ BBOBFunctions.F1.x_opt[1:2]

#

using Gadfly
import Colors.@colorant_str

function plot_trace(f,trace)

    r = 6
    t = hcat(trace.ps...)
    
    c, n, nline = Geom.contour(levels=logspace(-6,10,40)), 800, 1000
    
    p1 = plot(
        layer(x=[f.x_opt[1]],y=[f.x_opt[2]],Geom.point,Theme(default_color=colorant"red")),
        layer(
            z=(x,y)->f([x,y])-f.f_opt, x=linspace(-r,r,n), y=linspace(-r,r,n), 
            c,
        ),
        layer(x=t[1,:],y=t[2,:],Geom.line,Theme(default_color=colorant"gray")),
        Coord.cartesian(xmin=-r,xmax=r,ymin=-r,ymax=r),
        Guide.title(string(f))
    )
    p1
end

plot_trace(F,gd.trace)

##