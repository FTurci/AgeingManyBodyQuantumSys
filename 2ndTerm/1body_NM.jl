module NM_measures_1body

# Export all relevant functions and types so they are accessible in your notebook
export base_params, set_P, spectral_function, thermofield_double, 
       chain_map, H_bath, H_tot, prepare_corrs, evolve_corrs, 
       spin_operators, matrix_operators, matrix_log, map_to_principal, 
       ρ_to_Λ, Λ_to_ρ, calculate_ρ_using_G, extract_maps, calculate_BLP, calculate_RHP,
       calculate_Mpemba, calculate_TDFP, calculate_oCP, calculate_iCP, calculate_Signalling,
       extract_process_tensor, calculate_PT_oCP

using LinearAlgebra
using PolyChaos
using QuadGK
using Plots
using ProgressMeter
using Kronecker
using JLD2
using SparseArrays
using Contour

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
    T ::Float64                      #final time
    base_params() = new()
end

function set_P()
    """initialises the base params P to some default settings"""
    P = base_params();
    P.spec_fun = "ellipse"
    P.v = 100.0
    P.N_L = 50
    P.N_R = 50
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
    P.T = 100.0
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
        norm = quadgk(J, -D, D)[1]           
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
    """creates initial correlation matrix empty, central system matrix needs to be added in"""
    (;N_L, N_R) = P
    Csys = complex([0.0 0.0 ; 0.0 0.0])
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
        C0 = cat(Csys, C_R; dims=(1,2))
        return C0
    end

    C_L = make_bath_corr(N_L)
    C_R = make_bath_corr(N_R)

    C0 = cat(reverse(C_L), Csys, C_R; dims=(1,2))

    return C0
end

function evolve_corrs(C0, H, P, T)
    (;dt) = P
    C_curr = Matrix(C0)
    H = Matrix(H)
    
    step_max = Int(round(T/dt)) + 1
    
    U_dt = exp(-im * dt * H) 
    U_dt_dag = U_dt'
    
    Cs = Vector{Any}(undef, step_max)
    Cs[1] = C_curr
    step = 2

    while step <= step_max
        C_curr = U_dt * C_curr * U_dt_dag
        Cs[step] = C_curr
        """
        if (step % 100 == 0) && (step >= step_min)
            if length(Cs) > 100
                nSys = [real(M[qS, qS]) for M in Cs[end-100:end]]
                min_val, max_val = extrema(nSys)
                settled = (max_val - min_val < 1e-5)
            end
        end
        """
        step += 1
        #if step >= step_max; break; end #break when end of chains perturb
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
            # Corrected to reflect Ancilla (i_a) x System (i_s) basis
            Λ[(i_s-1)*d + j_s, (i_a-1)*d + j_a] = 
                conj(d*ρ[(i_a-1)*d + i_s, (j_a-1)*d + j_s])
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
    for i_s=1:d, j_s=1:d
        for i_a=1:d, j_a=1:d
            row_L = (i_s-1)*d + j_s
            col_L = (i_a-1)*d + j_a
            
            # Corrected to reflect Ancilla x System basis
            row_R = (i_a-1)*d + i_s
            col_R = (j_a-1)*d + j_s
            
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
    
    """
    # --- Added Eigenvalue Regularization ---
    F_G = eigen(G)
    # Clamp the real parts to avoid exactly 0.0 or 1.0
    reg_evals = [clamp(real(v), 1e-12, 1.0 - 1e-12) + im*imag(v) for v in F_G.values]
    G = F_G.vectors * Diagonal(reg_evals) * inv(F_G.vectors)
    # ---------------------------------------
    """
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

function extract_maps(P, C0_input, T; num=3)
    (;dt, N_L) = P
    qA = 2*N_L + 1 
    qS = 2*N_L + 2 

    C0 = copy(C0_input)
    
    #severing system-bath correlations
    C0[qA:qS, 1:qA-1] .= 0.0
    C0[1:qA-1, qA:qS] .= 0.0
    C0[qA:qS, qS+1:end] .= 0.0
    C0[qS+1:end, qA:qS] .= 0.0

    #CJ isomorphism
    C0[qA:qS, qA:qS] .= 0.5
    
    J_L = spectral_function(P, "L")
    J_R = spectral_function(P, "R")
    H = H_tot(J_L, J_R, P)
    
    Cs = evolve_corrs(C0, H, P, T)[2:end]

    if num==1
        Λ_vec = [calculate_ρ_using_G(Cs[i], qS, qA) for i in 1:length(Cs)]
        return Λ_vec
    elseif num==2 
        Λ_vec = [calculate_ρ_using_G(Cs[i], qS, qA) for i in 1:length(Cs)]
        Λ_inv = pinv.(Λ_vec[1:end-1])
        V_vec = Λ_vec[2:end] .* Λ_inv #vector of intermediate maps
        return Λ_vec, V_vec
    else
        Λ_vec = [calculate_ρ_using_G(Cs[i], qS, qA) for i in 1:length(Cs)]
        Λ_inv = pinv.(Λ_vec[1:end-1])
        V_vec = Λ_vec[2:end] .* Λ_inv #vector of intermediate maps
        L_vec = ((Λ_vec[2:end]-Λ_vec[1:end-1])./dt) .* Λ_inv #vector of generators
        return Λ_vec, V_vec, L_vec
    end
end

function calculate_BLP(P; plotting=false)
    (;Γ_R, β_R, μ_R, dt, N_L, T) = P
    
    qS = 2*N_L + 2 

    J_L = spectral_function(P, "L")
    J_R = spectral_function(P, "R")
    
    H = H_tot(J_L, J_R, P)
    C0_e = prepare_corrs(P)
    C0_f = prepare_corrs(P)
    C0_f[qS,qS] = 1.0
    Cs_e = evolve_corrs(C0_e, H, P, T)
    Cs_f = evolve_corrs(C0_f, H, P, T)

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
function calculate_RHP(P; plotting=false)
    (;dt, T, N_L) = P
    
    C0 = prepare_corrs(P)
    Λ_vec, V_vec, L_vec = extract_maps(P, C0, T)
    
    times = [i*dt for i in 1:length(V_vec)]
    rhp_accum = zeros(length(V_vec))
    rhp_step = zeros(length(V_vec))
    current_rhp = 0.0
    
    if plotting==true #only need generator eigenvalues for plotting
        L_evals = [Float64[] for _ in 1:4]
        for (i, L) in enumerate(L_vec)
            # Extract and sort eigenvalues for generator
            evals_current = eigen(L).values
            sort!(evals_current, by = x -> (real(x), imag(x)))
            evals_real = real.(evals_current)
            for j in 1:4
                push!(L_evals[j], evals_real[j])
            end
        end
    end

    for (i, V) in enumerate(V_vec)
        # Choi State Trace Norm Calculation
        ρ_choi = Λ_to_ρ(V, 1)
        trace_norm = sum(svdvals(ρ_choi))

        diff = trace_norm - 1.0
        rhp_step[i] = diff / dt

        current_rhp += diff 
        rhp_accum[i] = current_rhp    
    end
    
    final_rhp = rhp_accum[end]
    println("Final RHP Measure: ", final_rhp)

    # --- PLOTTING ---
    if plotting==true
        
        p1 = plot(times, rhp_accum, xlabel="Time", ylabel="\$\\mathcal{N}_{RHP}(t)\$", legend=false, lw=2, dpi=500)
        
        p2 = plot(times, rhp_step, label="", xlabel="Time", lw=2,
            ylabel="\$||\\mathbb{I} \\otimes \\Lambda(t+\\delta t, t) P^+ || - 1 \$", legend=false, dpi=500)
        
            
        p3 = plot(xlabel="Time", ylabel="\$Re[\\mathcal{L}_i (t)]\$", legend=:bottomright, dpi=500)
        
        for j in 1:4
            plot!(p3, times, L_evals[j], label="", lw=2)
        end

        # Mark convergence time
        """
        if converge_time > 0.0
            vline!(p1, [converge_time], line=:dash, color=:black)
            vline!(p2, [converge_time], line=:dash, color=:black)
            vline!(p3, [converge_time], line=:dash, color=:black, label="Generator convergence")
        end
        """
        g = P.Γ_R; mu = P.μ_R; beta = P.β_R;
        p_combined = plot(p1, p2, p3, plot_title="\$\\beta=$beta, \\mu=$mu, \\Gamma=$g\$",
                        layout=(3, 1), size=(800, 800))
        display(p_combined)
    end

    return final_rhp
end

function calculate_TDFP(P; plotting=false)
    (; dt, T, N_L) = P
    
    C0 = prepare_corrs(P)
    Λ_vec, V_vec = extract_maps(P, C0, T, num=2)
    times = [i*dt for i in 1:length(Λ_vec)]

    V_TDFP = zeros(length(times) - 1)
    for (i,V) in enumerate(V_vec)
        F = eigen(V)
        idx = argmin(abs.(abs.(F.values) .- 1.0)) #eigenvalue closest to 1
        ρ_TDFP = F.vectors[:, idx] #corresponding eigenvector
        V_TDFP[i] = real(ρ_TDFP[end]) / real(sum(ρ_TDFP)) #population of the time-dependent fixed point
    end
    extremes = [extrema(V_TDFP[i-19:i]) for i in 20:(length(V_TDFP))]
    mem_idx = findfirst(i -> extremes[i][2] - extremes[i][1] < 1e-3, 1:length(extremes)) 
    mem_time = (mem_idx == nothing) ? NaN : times[mem_idx+19] 
    if plotting == true
        Λ_TDFP = zeros(length(times))
        for (i,Λ) in enumerate(Λ_vec)
            F = eigen(Λ)
            idx = argmin(abs.(abs.(F.values) .- 1.0)) #eigenvalue closest to 1
            ρ_TDFP = F.vectors[:, idx] #corresponding eigenvector
            Λ_TDFP[i] = real(ρ_TDFP[end]) / real(sum(ρ_TDFP)) #population of the time-dependent fixed point
        end
        p = plot(times, Λ_TDFP, xlabel="time", ylabel="\$p_{TDFP}\$", label="\$\\Lambda\$", lw=2, dpi=200)
        plot!(p, times[1:end-1], V_TDFP, label="\$\\mathcal{V}\$", lw=2)
        vline!(p, [mem_time], linestyle=:dot, c=:black, label="\$\\tau^\\mathcal{L}_m\$")
        display(p)
    end
    
    return mem_time, mem_idx

end

function calculate_Mpemba(P, mem_time; plotting=false)
    """Calculates non-Markovian Mpemba state based on a given memory time"""
    (; dt, T, N_L) = P
    
    C0 = prepare_corrs(P)
    Λ_vec, V_vec = extract_maps(P, C0, T, num=2)
    times = [i*dt for i in 1:length(Λ_vec)]
    idx = argmin(abs.(times .- mem_time)) #memory time index

    S = Λ_vec[idx] #slippage operator
    S_inv  = pinv(S) #inverse slippage

    F = eigen(V_vec[end]) #TDFP converges faster for V than Λ
    idx = argmin(abs.(abs.(F.values) .- 1.0)) #eigenvalue closest to 1
    ρ_inf = F.vectors[:, idx] #steady state
    pinf = real(ρ_inf[end]/sum(ρ_inf)) #population of steady state
    ρ_f = S_inv * ρ_inf #Mpemba state
    ρ_f = ρ_f / tr(reshape(ρ_f, 2, 2))
    pf = real(ρ_f[end]) #population of Mpemba state
    println("Mpemba state is $pf, Steady state is $pinf.")

    return pf, pinf
end

function calculate_oCP(P; plotting=false)
    (;dt, T, N_L) = P
    qA = 2*N_L + 1
    qS = 2*N_L + 2

    J_L = spectral_function(P, "L")
    J_R = spectral_function(P, "R")
    C0 = prepare_corrs(P)
    
    C0[qA:qS, qA:qS] .= 0.5 
    
    H = H_tot(J_L, J_R, P)
    Cs = evolve_corrs(C0, H, P, T)[2:end]
    Λ_vec = extract_maps(P, C0, 2*T, num=1)
    
    times = [i*dt for i in 1:length(Cs)]
    len = length(times)
    tracedist = zeros(len, len)
    CPviolation = zeros(len, len)
    @showprogress for i in 1:len
        Λs = Λ_vec[i]
        C = Cs[i]
        # C is copied and decoupled inside extract_maps
        Φ_vec = extract_maps(P, C, T, num=1)
        
        for j in 1:len
            # Match the correct future time step in the overarching Λ_vec
            Λt = Λ_vec[i + j]
            Φ = Φ_vec[j] 
            
            composite_map = Φ * Λs
            ρ_Λ = Λ_to_ρ(Λt, 1)
            ρ_ΦΛ = Λ_to_ρ(composite_map, 1)
            tracedist[j, i] = 0.5 * sum(svdvals(ρ_Λ - ρ_ΦΛ))#trace distance between choi states
            ρ_Φ = Λ_to_ρ(Φ, 1)
            CPviolation[j, i] = sum(svdvals(ρ_Φ)) - 1.0 #trace norm of choi state for Phi
        end
    end

    if plotting==true
        p = heatmap(times, times, tracedist, xlabel="\$s\$", ylabel="\$t-s\$",
                    title="\$\\frac{1}{2}||\\rho_{t:0} - \\rho_{t:s:0}||_1\$", dpi=400)
        c = contours(times, times, tracedist, 6)
        for cl in levels(c)
            for line in lines(cl)
                xs, ys = coordinates(line)
                plot!(p, ys, xs, c=:white, label="")
            end
        end
        display(p)

        p = heatmap(times, times, CPviolation, xlabel="\$s\$", ylabel="\$t-s\$",
                    title="\$||\\rho_{t:s}|| - 1\$", dpi=400)
        display(p)
    end

    return Matrix(tracedist), Matrix(CPviolation), times

end

function calculate_iCP(P; plotting=false)
    (;dt, T, N_L) = P
    
    C0 = prepare_corrs(P)
    Λ_vec = extract_maps(P, C0, T, num=1)
    times = [i*dt for i in 1:length(Λ_vec)]
    len = length(times)
    tracenorm = zeros(len, len)
    @showprogress for i in 1:len-1
        Λs = Λ_vec[i]
        Λs_inv = pinv(Λs)
        for j in 1:len-i
            # Match the correct future time step in the overarching Λ_vec
            Λt = Λ_vec[i + j]
            Λts = Λt * Λs_inv 
            ρ_choi = Λ_to_ρ(Λts, 1)
            
            tracenorm[i, i + j] = sum(svdvals(ρ_choi)) - 1.0
        end
    end

    if plotting==true
        p = heatmap(times, times, tracenorm, xlabel="t", ylabel="s",
                    title="\$||\\mathhbb{I}\\otimes\\Lambda(t,s)P^+||_1\$", dpi=400)
        c = contours(times, times, tracenorm, 3)
        for cl in levels(c)
            for line in lines(cl)
                xs, ys = coordinates(line)
                plot!(p, ys, xs, c=:white, label="")
            end
        end
        display(p)
    end

    return Matrix(tracenorm), times
end

function calculate_Signalling(P; plotting=false)
    (;dt, T, N_L) = P
    qA = 2*N_L + 1
    qS = 2*N_L + 2

    J_L = spectral_function(P, "L")
    J_R = spectral_function(P, "R")
    H = H_tot(J_L, J_R, P)

    C0_e = prepare_corrs(P) #empty initial state
    C0_e[qS,qS] = 0.0 
    C0_f = prepare_corrs(P) #filled initial state
    C0_f[qS,qS] = 1.0 

    Cs_e = evolve_corrs(C0_e, H, P, T)[2:end]
    Cs_f = evolve_corrs(C0_f, H, P, T)[2:end]
    
    times = collect(dt:dt:T)
    len = length(times)
    tracedist = zeros(len, len)
    @showprogress for i in 1:len
        C_e = Cs_e[i]
        C_f = Cs_f[i]
        # C is copied and decoupled inside extract_maps
        Φ_vec_e = extract_maps(P, C_e, T, num=1)
        Φ_vec_f = extract_maps(P, C_f, T, num=1)
        for j in 1:len
            Φ_e = Φ_vec_e[j]
            Φ_f = Φ_vec_f[j]

            
            tracedist[j, i] = 0.5 * sum(svdvals(Φ_f - Φ_e))#trace distance between choi states
        end
    end

    if plotting==true
        p = heatmap(times, times, tracedist, xlabel="\$s\$", ylabel="\$t-s\$",
                    title="\$\\frac{1}{2}||L_f (t,s) - L_e (t,s)||\$", dpi=400)
        c = contours(times, times, tracedist, 3)
        for cl in levels(c)
            for line in lines(cl)
                xs, ys = coordinates(line)
                plot!(p, ys, xs, c=:white, label="")
            end
        end
        display(p)
    end

    return Matrix(tracedist), times

end

########################################
########################################
function extract_process_tensor(P, C0_input, s_time, t_time)
    (;dt, N_L, N_R) = P
    
    s_steps = Int(round(s_time / dt))
    t_steps = Int(round((t_time - s_time) / dt))
    
    N_base = 2*N_L + 2*N_R + 2
    qA = 2*N_L + 1
    qS = 2*N_L + 2
    
    N_tot = N_base + 2
    idx_S_minus = N_base + 1
    idx_As = N_base + 2
    
    C_ext = zeros(ComplexF64, N_tot, N_tot)
    C_ext[1:N_base, 1:N_base] = copy(C0_input)
    C_ext[qA:qS, qA:qS] .= 0.5
    
    # Create a parallel matrix that will NOT undergo the intervention at time s
    C_actual = copy(C_ext)
    
    J_L = spectral_function(P, "L")
    J_R = spectral_function(P, "R")
    H_base = H_tot(J_L, J_R, P)
    
    H_ext = zeros(ComplexF64, N_tot, N_tot)
    H_ext[1:N_base, 1:N_base] = H_base
    
    U_ext = exp(-im * dt * H_ext)
    U_ext_dag = U_ext'
    
    # 2. Evolve from 0 to s
    for _ in 1:s_steps
        C_ext = U_ext * C_ext * U_ext_dag
        C_actual = U_ext * C_actual * U_ext_dag # Un-intervened evolution
    end
    
    # 3. Intervention at time s (applied to C_ext ONLY)
    C_ext[idx_S_minus, :] = C_ext[qS, :]
    C_ext[:, idx_S_minus] = C_ext[:, qS]
    C_ext[idx_S_minus, idx_S_minus] = C_ext[qS, qS]
    
    C_ext[qS, :] .= 0.0
    C_ext[:, qS] .= 0.0
    
    C_ext[idx_As, idx_As] = 0.5
    C_ext[qS, qS] = 0.5
    C_ext[idx_As, qS] = 0.5
    C_ext[qS, idx_As] = 0.5
    
    # 4. Evolve from s to t
    for _ in 1:t_steps
        C_ext = U_ext * C_ext * U_ext_dag
        C_actual = U_ext * C_actual * U_ext_dag # Continues un-intervened
    end
    
    # 5. Extract the covariance matrices
    pt_indices = [qA, idx_S_minus, idx_As, qS]
    G_PT = transpose(C_ext[pt_indices, pt_indices])
    
    actual_indices = [qA, qS]
    G_tr_actual = transpose(C_actual[actual_indices, actual_indices])
    
    return G_PT, G_tr_actual
end

function calculate_PT_choi_using_G(G, num_modes)
    Id = Diagonal(ones(Float64, num_modes))

    α = matrix_log(G * pinv(Id - G))

    A = complex(zeros(2^num_modes, 2^num_modes))
    cdag_mat, c_mat = matrix_operators(num_modes)
    _, Sp, Sm, _ = spin_operators(num_modes)
    
    for (i, creator_i) in enumerate(cdag_mat)
        for (j, annihilator_j) in enumerate(c_mat)
            corr_op = Matrix(creator_i) * Matrix(annihilator_j)
            A += α[i, j] * corr_op
        end
    end

    ρ_choi = det(Id - G) * exp(A)

    # Particle-Hole transform must be applied to BOTH ancillas
    # Ancillas are at index 1 (A_r) and index 3 (A_s)
    PH_gate = (Sp[1] + Sm[1]) * (Sp[3] + Sm[3]) 
    ρ_choi = PH_gate * ρ_choi * PH_gate'
    
    return ρ_choi
end

function calculate_PT_choi_using_G_modified(G, num_modes, ancilla_indices)
    Id = Diagonal(ones(Float64, num_modes))
    
    α = matrix_log(G * pinv(Id - G))

    A = complex(zeros(2^num_modes, 2^num_modes))
    cdag_mat, c_mat = matrix_operators(num_modes)
    _, Sp, Sm, _ = spin_operators(num_modes)
    
    for (i, creator_i) in enumerate(cdag_mat)
        for (j, annihilator_j) in enumerate(c_mat)
            corr_op = Matrix(creator_i) * Matrix(annihilator_j)
            A += α[i, j] * corr_op
        end
    end

    ρ_choi = det(Id - G) * exp(A)

    # Dynamic Particle-Hole transform
    PH_gate = complex(Matrix(I, 2^num_modes, 2^num_modes))
    for idx in ancilla_indices
        PH_gate = PH_gate * (Sp[idx] + Sm[idx])
    end
    
    ρ_choi = PH_gate * ρ_choi * PH_gate'
    
    return ρ_choi
end

function partial_trace(ρ::Matrix{ComplexF64}, keep_modes::Vector{Int}, dims::Vector{Int})
    N = length(dims)
    
    # Map standard Kronecker modes (1=slowest) to Julia column-major tensor dimensions (1=fastest)
    tensor_keep = sort([N - m + 1 for m in keep_modes])
    tensor_drop = sort([N - m + 1 for m in setdiff(1:N, keep_modes)])
    
    # Reverse dims to match the inverted tensor dimensions
    rev_dims = reverse(dims)
    
    # Reshape into a 2N-dimensional tensor for bra and ket indices
    tensor_dims = Tuple(vcat(rev_dims, rev_dims))
    ρ_tensor = reshape(ρ, tensor_dims)
    
    # Permute the kept modes to the front
    perm = vcat(tensor_keep, tensor_keep .+ N, tensor_drop, tensor_drop .+ N)
    ρ_perm = permutedims(ρ_tensor, perm)
    
    # Flatten into a block matrix separating kept and dropped Hilbert spaces
    dim_keep = prod(rev_dims[tensor_keep])
    dim_drop = prod(rev_dims[tensor_drop])
    ρ_grouped = reshape(ρ_perm, (dim_keep, dim_keep, dim_drop, dim_drop))
    
    # Perform the trace over the dropped subsystem
    ρ_out = zeros(ComplexF64, dim_keep, dim_keep)
    for i in 1:dim_drop
        ρ_out += ρ_grouped[:, :, i, i]
    end
    
    return ρ_out
end

function calculate_PT_oCP(G_PT::AbstractMatrix{ComplexF64}, G_tr_actual::AbstractMatrix{ComplexF64})
    d = 2 

    # -------------------------------------------------------------------
    # Step 1: Extract Intermediate Choi States via Covariance Marginals
    # -------------------------------------------------------------------
    G_ts = G_PT[[3, 4], [3, 4]]
    L_ts = d * calculate_PT_choi_using_G_modified(G_ts, 2, [1])
    
    G_sr = G_PT[[1, 2], [1, 2]]
    L_sr = d * calculate_PT_choi_using_G_modified(G_sr, 2, [1])

    # -------------------------------------------------------------------
    # Step 2: Measure Conditional Signalling 
    # -------------------------------------------------------------------
    G_marg = G_PT[[1, 3, 4], [1, 3, 4]]
    T_marg = 4.0 * calculate_PT_choi_using_G_modified(G_marg, 3, [1, 2])
    
    Ideal_T_marg = kron(Matrix(I, 2, 2), L_ts)
    N_signalling = 0.5 * sum(svdvals(T_marg - Ideal_T_marg))
    
    # -------------------------------------------------------------------
    # Step 3: Extract the Actual Channel L_{t:r}
    # -------------------------------------------------------------------
    # We bypass the 16x16 tensor contraction by generating the Choi state 
    # directly from the un-intervened covariance matrix.
    L_tr_actual = d * calculate_PT_choi_using_G_modified(G_tr_actual, 2, [1])

    # -------------------------------------------------------------------
    # Step 4: Concatenate Intermediate Maps & Measure Divisibility
    # -------------------------------------------------------------------
    Λ_ts = ρ_to_Λ(L_ts ./ d, 1)
    Λ_sr = ρ_to_Λ(L_sr ./ d, 1)
    
    Λ_concatenated = Λ_ts * Λ_sr
    
    # Map back to Choi state and multiply by d
    L_tr_concatenated = d .* Λ_to_ρ(Λ_concatenated, 1)

    # Calculate violation of Eq. 8 [cite: 228]
    N_divisibility = 0.5 * sum(svdvals(L_tr_actual - L_tr_concatenated))
    
    return N_signalling, N_divisibility
end

end