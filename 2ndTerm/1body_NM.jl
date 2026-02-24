module NM_measures_1body

# Export all relevant functions and types so they are accessible in your notebook
export base_params, set_P, spectral_function, thermofield_double, 
       chain_map, H_bath, H_tot, prepare_corrs, evolve_corrs, 
       spin_operators, matrix_operators, matrix_log, map_to_principal, 
       ρ_to_Λ, Λ_to_ρ, calculate_ρ_using_G, calculate_RHP, calculate_BLP, 
       calculate_mpemba

using LinearAlgebra
using PolyChaos
using QuadGK
using Plots
using ProgressMeter
using Kronecker
using JLD2
using SparseArrays

mutable struct base_params
    spec_fun ::AbstractString        #type of spectral function
    v ::Float64                      #band edge smoothing parameter
    N_L ::Int64                      #Number of left bath sites
    N_R ::Int64                      #Number of right bath sites
    β_L ::Float64                    #inverse temperature of left bath
    β_R ::Float64                    #inverse temperature of right bath
    μ_L ::Float64                    #chemical potential of left Bath
    μ_R ::Float64                    #chemical potential of right Bath
    D_L ::Float64                    #left bath bandwidth
    D_R ::Float64                    #Right bath bandwidth
    Γ_L ::Float64                    #overall left bath coupling strength
    Γ_R ::Float64                    #overall right bath coupling strength
    E_sys ::Float64                  #system site energy
    dt ::Float64                     #timestep length
    init_occ ::Float64               #initial occupation

    base_params() = new()
end

function set_P()
    """initialises the base params P to some default settings"""
    P = base_params();
    P.spec_fun = "ellipse"
    P.v = 100.0
    P.N_L = 100
    P.N_R = 100
    P.β_L = 20.0
    P.β_R = 20.0
    P.μ_L = 0.0
    P.μ_R = 0.0
    P.D_L = 1.0
    P.D_R = 1.0
    P.Γ_L = 0.01
    P.Γ_R = 0.01
    P.E_sys = 0.0
    P.dt = 0.1
    return P
end

function spectral_function(P, side)
    (;spec_fun, v, D_L, D_R, Γ_L, Γ_R) = P

    if side == "L"
        D = D_L
        g = Γ_L
    else
        D = D_R
        g = Γ_R
    end

    """creates spectral functions for a bath"""
    inband = x -> (-D <= x <= D)
    
    if spec_fun == "smoothed box"
        J = x -> 1.0 / ((2pi*(1.0+exp(v*(x-D)))*(1.0+exp(-v*(x+D)))))
        norm = quadgk(J, -D-0.5, D+0.5)[1]           
        Jnorm = x -> g * J(x) / norm 

    elseif spec_fun == "ellipse"
        J = x -> inband(x) ? sqrt(1 - (x/D)^2) : 0.0
        norm = quadgk(J, -1.2D, 1.2D)[1]           
        Jnorm = x -> g*D/pi * J(x) / norm

    elseif spec_fun == "smoothed ellipse"
        η = 1.0 / v  
        J = function(x)
            z = x/D + im * η
            return real(sqrt(complex(1.0 - z^2)))
        end
        norm = quadgk(J, -3D, 3D)[1]           
        Jnorm = x -> g*D/pi * J(x) / norm

    elseif spec_fun == "ohmic"
        J = x -> inband(x) ? abs(x) : 0.0
        norm = quadgk(J, -D, D)[1]           
        Jnorm = x -> g*D/pi * J(x) / norm

    elseif spec_fun == "lorentzian"
        J = x -> 1/(1 + (x/D)^2)
        norm = quadgk(J, -10D, 10D)[1]            
        Jnorm = x -> g*D/pi * J(x) / norm

    else
        error("spectral function type not recognized")
    end
    
    return Jnorm
end

function thermofield_double(J, beta::Float64, mu::Float64)
    """thermofield purification using fermi function ancilla"""
    fermi(k) = 1/(1 + exp(beta*k - beta*mu))
    J1 = w -> J(w) * fermi(w) #filled mode spectral density
    J2 = w -> J(w) * (1 - fermi(w)) #empty mode spectral density
    return J1, J2
end

function chain_map(J, N::Int64, D::Float64, P)
    """calculates family of monic orthogonal polynomials w.r.t the measure J(x) up to the Nth term."""
    (;spec_fun) = P
    if spec_fun == "smoothed box"
        supp = (-D-0.5, D+0.5)
    elseif spec_fun == "lorentzian"
        supp = (-10D, 10D)
    elseif spec_fun == "smoothed ellipse"
        supp = (-3D, 3D)
    else
        supp = (-D, D)
    end

    meas = Measure("bath", J, supp, false, Dict())
    ortho_poly = OrthoPoly("bath_op", N, meas; Nquad=100000)   
    chain = coeffs(ortho_poly)                                  
    E = chain[1:N,1] #site energies
    h = sqrt.(chain[1:N,2]) #site hoppings
    return E, h
end

function H_bath(J, P, side)
    (;N_L, N_R, D_L, D_R, β_L, β_R, μ_L, μ_R) = P
    if side == "L"
        N = N_L
        D = D_L
        beta = β_L
        mu = μ_L
    else
        N = N_R
        D = D_R
        beta = β_R
        mu = μ_R
    end
    """makes Hamiltonian for single bath in chain basis, interleaved filled and empty chain"""
    H = zeros(ComplexF64, 2N, 2N)
    J1, J2 = thermofield_double(J, beta, mu)
    E1, h1 = chain_map(J1, N, D, P) #filled chain
    E2, h2 = chain_map(J2, N, D, P) #empty chain

    g1 = h1[1] #filled chain to system coupling
    g2 = h2[1] #empty chain to system coupling

    t1 = h1[2:end] #chain hoppings
    t2 = h2[2:end] 

    for n in 1:N
        H[2n-1, 2n-1] = E1[n]
        H[2n, 2n] = E2[n]
    end
    
    for n in 1:N-1
        H[2n-1,2n+1] = t1[n]
        H[2n+1,2n-1] = t1[n]
        H[2n, 2n+2] = t2[n]
        H[2n+2, 2n] = t2[n]
    end

    return H, g1, g2
end

function H_tot(J_L, J_R, P)
    """full Hamiltonian for baths + system + ancilla"""
    (;N_L, E_sys) = P

    if N_L == 0 
        H_R, g1_R, g2_R = H_bath(J_R, P, "R")
        H_sys = [0.0 0.0 ; 0.0 E_sys] 
        H = cat(H_sys, H_R; dims=(1,2))
        H[2, 3] = g1_R
        H[3, 2] = g1_R
        H[2, 4] = g2_R
        H[4, 2] = g2_R
        return Matrix(H)
    end

    H_L, g1_L, g2_L = H_bath(J_L, P, "L")
    H_R, g1_R, g2_R = H_bath(J_R, P, "R")
    H_sys = [0.0 0.0 ; 0.0 E_sys]
    
    H = cat(reverse(H_L), H_sys, H_R; dims=(1,2))

    # Connection indices
    sys_idx = 2*N_L + 2
    
    H[sys_idx, 2N_L] = g1_L
    H[2N_L, sys_idx] = g1_L
    H[sys_idx, 2N_L - 1] = g2_L
    H[2N_L - 1, sys_idx] = g2_L

    H[sys_idx, 2N_L + 3] = g1_R
    H[2N_L + 3, sys_idx] = g1_R
    H[sys_idx, 2N_L + 4] = g2_R
    H[2N_L + 4, sys_idx] = g2_R

    return Matrix(H)
end

function prepare_corrs(P)
    """creates initial correlation matrix"""
    (;N_L, N_R) = P

    function make_bath_corr(N)
        C = zeros(ComplexF64, 2N, 2N)
        for n in 1:2N
            if isodd(n) 
                C[n,n] = 1.0 
            else 
                C[n,n] = 0.0 
            end
        end
        return C
    end

    if N_L == 0
        C_R = make_bath_corr(N_R)
        C_emp = complex([0.0 0.0 ; 0.0 0.0]) 
        C_full = complex([0.0 0.0 ; 0.0 1.0])
        C_CJ = complex([0.5 0.5 ; 0.5 0.5]) 

        C0_emp = cat(C_emp, C_R; dims=(1,2))
        C0_full = cat(C_full, C_R; dims=(1,2))
        C0_CJ = cat(C_CJ, C_R; dims=(1,2))
        return C0_emp, C0_full, C0_CJ
    end

    C_L = make_bath_corr(N_L)
    C_R = make_bath_corr(N_R)

    C_emp = complex([0.0 0.0 ; 0.0 0.0]) 
    C_full = complex([0.0 0.0 ; 0.0 1.0]) 
    C_CJ = complex([0.5 0.5 ; 0.5 0.5]) 

    C0_emp = cat(reverse(C_L), C_emp, C_R; dims=(1,2))
    C0_full = cat(reverse(C_L), C_full, C_R; dims=(1,2))
    C0_CJ = cat(reverse(C_L), C_CJ, C_R; dims=(1,2))

    return C0_emp, C0_full, C0_CJ
end

function evolve_corrs(C0, H, P)
    (;N_L, N_R, D_L, D_R, dt) = P
    Cs = Vector{Array{ComplexF64}}(undef, 0)
    C_curr = Matrix(C0)
    H = Matrix(H)
    
    step_min = 1000
    step_max = 2000
    """
    if N_L >= N_R
        step_max = round(Int, N_L/(dt*max(D_L, D_R))) 
    else
        step_max = round(Int, N_R/(dt*max(D_L, D_R)))
    end
    """
    U_dt = exp(-im * dt * H) 
    U_dt_dag = U_dt'
    
    push!(Cs, C_curr)
    
    step = 1
    settled = false
    
    qS = 2*N_L + 2 #system index

    while !settled
        C_curr = U_dt * C_curr * U_dt_dag
        push!(Cs, C_curr)
        
        if (step % 100 == 0) && (step >= step_min)
            if length(Cs) > 100
                nSys = [real(M[qS, qS]) for M in Cs[end-100:end]]
                min_val, max_val = extrema(nSys)
                settled = (max_val - min_val < 1e-5)
            end
        end
        step += 1
        if step > step_max; break; end 
    end
    return Cs
end

# ==============================================================================
# Dynamical Map Extraction & RHP Calculation Functions
# ==============================================================================

function spin_operators(M)
    sp = spdiagm(2,2,1=>ones(1))
    sm = spdiagm(2,2,-1=>ones(1))
    sz = spdiagm(2,2,0=>[1;-1]);
    num = spdiagm(2,2,0=>[0;1])
    Sz = Vector{Any}(undef, M)
    Sp = Vector{Any}(undef, M)
    Sm = Vector{Any}(undef, M)
    Num = Vector{Any}(undef,M)
    for m=1:M
        Sz[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sz),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
        Sp[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sp),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
        Sm[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sm),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
        Num[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),num),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
    end
    return Sz,Sp,Sm,Num
end

function JW_string_mat(Sz,site,M)
    Z = 1.0*Matrix(I, 2^M, 2^M)
    for q=1:(site-1)
        Z = Z*Sz[q];
    end
    return Z
end

function matrix_operators(M)
    Sz,Sp,Sm,_ = spin_operators(M)
    cdag_mat = Vector{Any}(undef,M)
    c_mat = Vector{Any}(undef,M)
    for n=1:M
        Z = JW_string_mat(Sz,n,M)
        cdag_mat[n] = Z*Sm[n]
        c_mat[n]  = Z*Sp[n]
    end
    return cdag_mat,c_mat
end

function map_to_principal(z)
    im_ = imag(z)
    im_ = im_ - 2*π*floor((im_ + π)/(2*π))
    return Complex(real(z), im_)
end

function matrix_log(A)
    F = eigen(A)
    
    # Extract eigenvalues and apply the logarithm only to them
    log_evals = map_to_principal.(log.(Complex.(F.values)))
    
    # Construct the diagonal matrix AFTER the logarithm is applied
    principal_log_D = Diagonal(log_evals)
    
    V = F.vectors
    V_inv = inv(V)
    log_A = V * principal_log_D * V_inv
    
    return log_A
end

function ρ_to_Λ(ρ, Ns)
    """Rearranges the Choi state ρ to give the Dynamical Map Λ."""
    d = 2^Ns
    Λ = zeros(ComplexF64, d^2, d^2) 
    for i_s=1:d, j_s=1:d
        for i_a=1:d, j_a=1:d  
            Λ[(i_s-1)*d + j_s, (i_a-1)*d + j_a] = 
                conj(d*ρ[(i_s-1)*d + i_a, (j_s-1)*d + j_a])
        end
    end
    return Λ
end

function Λ_to_ρ(Λ, Ns)
    """
    Reverse operation of ρ_to_Λ. 
    Constructs the Choi matrix ρ of a map Λ.
    """
    d = 2^Ns
    ρ = zeros(ComplexF64, d^2, d^2)
    # The inverse relation derived from ρ_to_Λ
    for i_s=1:d, j_s=1:d
        for i_a=1:d, j_a=1:d
            row_L = (i_s-1)*d + j_s
            col_L = (i_a-1)*d + j_a
            
            row_R = (i_s-1)*d + i_a
            col_R = (j_s-1)*d + j_a
            
            ρ[row_R, col_R] = conj(Λ[row_L, col_L]) / d
        end
    end
    return ρ
end


function calculate_ρ_using_G(corr_full, qS, qA)
    Ns = 1 
    idxs = [qA, qS]
    
    # Transpose is necessary per the paper's definition of G
    G = transpose(corr_full[idxs, idxs])
    Id = Diagonal(ones(Float64, 2*Ns))    
    
    # --- Added Eigenvalue Regularization ---
    F_G = eigen(G)
    # Clamp the real parts to avoid exactly 0.0 or 1.0
    reg_evals = [clamp(real(v), 1e-10, 1.0 - 1e-10) + im*imag(v) for v in F_G.values]
    G = F_G.vectors * Diagonal(reg_evals) * inv(F_G.vectors)
    # ---------------------------------------

    α = matrix_log(G * pinv(Id - G))

    A = complex(zeros(2^(2*Ns), 2^(2*Ns)))
    cdag_mat, c_mat = matrix_operators(2*Ns)
    _, Sp, Sm, _ = spin_operators(2*Ns)
    
    for (i, creator_i) in enumerate(cdag_mat)
        for (j, annihilator_j) in enumerate(c_mat)
            corr_op = Matrix(creator_i) * Matrix(annihilator_j)
            A += α[i, j] * corr_op
        end
    end

    ρ = det(Id - G) * exp(A)

    # Particle-Hole transform
    PH_gate = Sp[1] + Sm[1] 
    ρ = PH_gate * ρ * PH_gate'
    
    Λ = ρ_to_Λ(ρ, Ns) 
    return Λ
end


function calculate_BLP(P; plotting=false)
    (;spec_fun, Γ_L, Γ_R, β_R, μ_R, D_L, D_R, dt, N_L) = P
    
    qS = 2*N_L + 2 

    J_L = spectral_function(P, "L")
    J_R = spectral_function(P, "R")
    
    H = H_tot(J_L, J_R, P)
    C0_e, C0_f, _ = prepare_corrs(P)
    Cs_e = evolve_corrs(C0_e, H, P)
    Cs_f = evolve_corrs(C0_f, H, P)

    rho1 = [real(C[qS,qS]) for C in Cs_f]
    rho2 = [real(C[qS,qS]) for C in Cs_e]
    l1 = length(rho1)
    l2 = length(rho2)

    if l1 < l2
        rho1 = vcat(rho1, rho1[end].*ones(l2-l1))
    elseif l1 > l2
        rho2 = vcat(rho2, rho2[end].*ones(l1-l2))
    end
    
    TD = 0.5 .* abs.(rho1 .- rho2) #single-particle trace distance
    sigma = TD[2:end] .- TD[1:end-1] #change in trace distance
    
    times = dt .* collect(0:length(rho1)-2)
    blp_accum = Float64[]
    for i in 1:length(rho1)-1
        sigma_current = sigma[1:i-1]
        blp_current = sum(sigma_current[sigma_current .>= 0.0]) #sum over all positive change in trace distance
        push!(blp_accum, blp_current)
    end
    blp_final = blp_accum[end]

    if plotting==true
        p1 = plot(times, rho1[1:end-1], xlabel="Time", ylabel="\$ <N> \$",legend=false, lw=2)
        plot!(p1, times, rho2[1:end-1], lw=2)
        p2 = plot(times, blp_accum, label="", xlabel="Time", ylabel="BLP Measure",
         title="\$\\Gamma=$Γ_R, \\beta=$β_R, \\mu=$μ_R\$", lw=3, dpi=400)
        p = plot(p1, p2, layout=(2,1), size=(800,600), dpi=400)
        display(p)
    end

    return blp_final
end

#test function
function calculate_RHP(P, min_diff)
    (;spec_fun, Γ_L, Γ_R, D_L, D_R, dt, N_L) = P
    
    qA = 2*N_L + 1 
    qS = 2*N_L + 2 

    println("Calculating Spectral Functions...")
    J_L = spectral_function(P, "L")
    J_R = spectral_function(P, "R")
    
    println("Constructing Hamiltonian and Evolving State...")
    H = H_tot(J_L, J_R, P)
    _, _, C0_CJ = prepare_corrs(P)
    Cs_CJ = evolve_corrs(C0_CJ, H, P)
    
    times = [i*dt for i in 0:length(Cs_CJ)-1]
    rhp_accum = Float64[]
    push!(rhp_accum, 0.0)
    current_rhp = 0.0

    # Accumulators for the 4 eigenvalues and singularity points
    evals_history = [Float64[] for _ in 1:4]
    singular_times = Float64[]
    
    # Convergence Tracking Variables
    eval_tol = 1e-4 #absolute tolerance
    steps_required = Int(10.0/dt)
    consecutive_success = 0
    converge_time = 0.0
    prev_evals_real = zeros(Float64, 4)
    diff_sequence  = Float64[]
    println("Extracting Exact RHP Measure and Generator Eigenvalues...")
    
    for i in 1:(length(Cs_CJ)-1)
        
        # Extract maps for current, next, and previous time steps
        Λ_t = calculate_ρ_using_G(Cs_CJ[i], qS, qA)
        Λ_next = calculate_ρ_using_G(Cs_CJ[i+1], qS, qA)

        # ======================================================================
        # 1. SINGULARITY TRACKING
        # ======================================================================
        
        # Track when the dynamical map loses full rank
        F_SVD = svd(Λ_t)
        if minimum(F_SVD.S) < 1e-10
            push!(singular_times, times[i])
        end

        # ======================================================================
        # 2. EXACT GENERATOR & CONVERGENCE EXTRACTION
        # ======================================================================
        
        # Compute exact matrix derivative dΛ/dt via central difference
        if i == 1
            dΛ_dt = (Λ_next - Λ_t) / dt
        else
            Λ_prev = calculate_ρ_using_G(Cs_CJ[i-1], qS, qA)
            dΛ_dt = (Λ_next - Λ_prev) / (2 * dt)
        end

        # Construct full generator superoperator L(t)
        # Tolerance applied to suppress singularity noise
        L_t = dΛ_dt * pinv(Λ_t, rtol=1e-10)

        # Extract and sort eigenvalues lexicographically
        evals_current = eigen(L_t).values
        sort!(evals_current, by = x -> (real(x), imag(x)))
        evals_real = real.(evals_current)
        
        for j in 1:4
            push!(evals_history[j], evals_real[j])
        end

    # Track Convergence of Real Eigenvalues
        max_eval_diff = maximum(abs.(evals_real .- prev_evals_real)) / dt
        if max_eval_diff < eval_tol
            consecutive_success += 1
            if consecutive_success == steps_required && converge_time == 0.0
                converge_time = times[i] - (steps_required * dt)
            end
        else
            consecutive_success = 0
        end

        prev_evals_real = evals_real

        # ======================================================================
        # 3. RHP MEASURE EXTRACTION (Exact Discrete Map Method)
        # ======================================================================
        
        # Calculate exact forward intermediate map
        V_exact = Λ_next * pinv(Λ_t, rtol=1e-10)
        
        # Choi State Trace Norm Calculation
        ρ_exact = Λ_to_ρ(V_exact, 1)
        ρ_exact = Hermitian(0.5 * (ρ_exact + ρ_exact'))
        
        trace_norm = sum(abs.(eigen(ρ_exact).values))
        diff = trace_norm - 1.0
        
        push!(diff_sequence, diff)
        if diff > min_diff
            current_rhp += diff 
        end
        push!(rhp_accum, current_rhp)
    end
    
    if converge_time > 0.0
        println("Generator convergence detected at t = $(converge_time)")
    else
        println("Generator did not fully converge.")
    end
    println("Final RHP Measure: ", current_rhp)
    println("Dynamical map was singular at $(length(singular_times)) measured time intervals.")
    
    # --- PLOTTING ---
    time_axis = times[1:length(rhp_accum)]
    
    p1 = plot(time_axis, rhp_accum, 
         title="Exact RHP Measure", 
         xlabel="Time", ylabel="RHP(t)", legend=false, linewidth=2)
         
    time_axis_evals = times[1:length(evals_history[1])]
    
    p2 = plot(title="Real Part of Generator Eigenvalues", 
              xlabel="Time", ylabel="Re(λ_i)", legend=:bottomright)
    
    for j in 1:4
        plot!(p2, time_axis_evals, evals_history[j], label="λ_$j", linewidth=2)
    end
    
    # Mark convergence time
    if converge_time > 0.0
        vline!(p1, [converge_time], line=:dash, color=:black, label="Convergence")
        vline!(p2, [converge_time], line=:dash, color=:black, label="Convergence")
    end

    # Overlay scatter points along the x-axis for instances of singularities
    if length(singular_times) > 0
        scatter!(p2, singular_times, zeros(length(singular_times)), color=:red, markersize=3, label="Singularities")
    end
    
    p3 = plot(diff_sequence)
    p_combined = plot(p1, p2, p3, layout=(3, 1), size=(800, 1000), dpi=400)
    display(p_combined)
    
    return current_rhp, rhp_accum
end

function calculate_mpemba(P)
    (; dt, N_L) = P
    
    qA = 2*N_L + 1 
    qS = 2*N_L + 2 

    println("Calculating Spectral Functions and Evolving...")
    J_L = spectral_function(P, "L")
    J_R = spectral_function(P, "R")
    H = H_tot(J_L, J_R, P)
    _, _, C0_CJ = prepare_corrs(P)
    Cs_CJ = evolve_corrs(C0_CJ, H, P)
    
    times = [i*dt for i in 0:length(Cs_CJ)-1]

    # Pre-calculate the entire sequence of dynamical maps
    Λ_vec = [calculate_ρ_using_G(Cs_CJ[i], qS, qA) for i in 1:length(Cs_CJ)]
    
    println("Locating Memory Time (τ_m) via Population Subspace...")
    
    
    return ρ_E_mat, ρ_SS_mat, delta_p
end

end