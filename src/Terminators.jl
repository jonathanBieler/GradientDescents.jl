    struct SimpleTerminator <: Terminator
    end
    init!(t::SimpleTerminator) = nothing
    
    
    function done(t::SimpleTerminator,f,g_t,trace) 
        
        vecnorm(g_t,Inf) < 1e-8 && return true
        maximum(abs(trace.p[i]-trace.p_previous[i]) for i=1:length(trace.p)) < 1e-32 && return true
        
        false
    end 