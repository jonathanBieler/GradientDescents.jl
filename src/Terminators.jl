    struct SimpleTerminator <: Terminator
    end
    init!(t::SimpleTerminator) = nothing
    done(t::SimpleTerminator,f,trace) = false