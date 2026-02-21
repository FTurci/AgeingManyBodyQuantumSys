module NM_measures_1body

"""functions for simulating single quantum dot connected to two thermal baths,
 specifically to calculate NM measures."""

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
        N_L ::Int64                      #Number of left bath sites
        N_R ::Int64                      #Number of right bath sites
        β_L ::Float64                    #inverse temperature of right bath
        β_R ::Float64                    #inverse temperature of left bath
        μ_L ::Float64                   #chemical potential of left Bath
        μ_R ::Float64                   #chemical potential of right Bath
        D_L ::Float64                   #left bath bandwidth
        D_R ::Float64                   #Right bath bandwidth
        Γ_L ::Float64                   #overall left bath coupling strength
        Γ_R ::Float64                   #overall right bath coupling strength
        E_sys ::Float64                 #system site energy
        dt ::Float64                    #timestep length

        base_params() = new()
    end

    function set_P()
        """initialises the base params P to soem default settings"""
        P = Base_params();
        P.spec_fun = "elliptical"
        P.N_L = 100
        P.N_R = 100
        P.β_L = 10
        P.β_R = 10
        P.μ_L = 0.0
        P.μ_R = 0.0
        P.D_L = 1.0
        P.D_R = 1.0
        P.Γ_L = 0.01
        P.Γ_R = 0.01
        P.E_sys = 0.0
        P.init_occ = 0.0
        return P
    end

    function spectral_function(input::AbstractString, D::Float64, g::Float64)
        """creates spectral functio(s) for a bath"""
        inband = x -> (-D <= x <= D)
        
        if input == "flat"
            J = x -> inband(x) ? 1/(2D) : 0.0
        elseif input == "elliptical"
            J = x -> inband(x) ? sqrt(1 - (x/D)^2) : 0.0
        elseif input == "ohmic"
            J = x -> inband(x) ? abs(x) : 0.0
        elseif input == "lorentzian"
            J = x -> 1/(1 + (x/D)^2)
        else
            error("spectral function type not recognized")
        end
        # normalization
        norm = quadgk(J, -D, D)[1]           
        Jnorm = x -> g*D/pi * J(x) / norm
        return Jnorm
    end

    function modify_spectral(J::Function, D, mean::Float64, std::Float64, amp::Float64)
        """modifies the spectral function by a gaussian perturbation"""
        inband = x -> (-D <= x <= D)
        Jmod = x -> inband(x) ? J(x) + (amp/sqrt(2pi*std^2) * exp(-((x-mean)^2)/(2*std^2))) : 0.0
        norm = quadgk(Jmod, -D, D)[1]           
        Jnorm = x -> g*D/pi * Jmod(x) / norm
        return Jnorm
    end

    function thermofield_double(J, beta::Float64, mu::Float64) #spectral function, inverse temp, chemical potential
        """thermofield purification using fermi function ancilla"""
        fermi(k) = 1/(1 + exp(beta*k - beta*mu))
        J1 = w -> J(w) * fermi(w) #filled mode spectral density
        J2 = w -> J(w) * (1 - fermi(w)) #empty mode spectral density
        return J1, J2
    end

    function chain_map(J, N::Int64, D::Float64)
        """calculates family of monic orthogonal polynomials w.r.t the measure J(x) up to the Nth term.
        returns the coefficients alpha and beta from the recurrence relation of the family."""
        supp = (-D, D)
        meas = Measure("bath", J, supp, false, Dict())
        ortho_poly = OrthoPoly("bath_op", N, meas; Nquad=100000)   
        chain = coeffs(ortho_poly)                                  
        E = chain[1:N,1] #site energies
        h = sqrt.(chain[1:N,2]) #site hoppings (first term is system hopping)
        return E, h
    end

    function H_bath(N::Int64, J, D::Float64, beta::Float64, mu::Float64)
    """makes Hamiltonian for single bath in chain basis, interleaved filled and empty chain"""

        H = zeros(ComplexF64, 2N, 2N)
        J1, J2 = thermofield_double(J, beta::Float64, mu::Float64) #thermofield doubling of spectral function
        E1, h1 = chain_map(J1, N::Int64, D::Float64) #filled chain
        E2, h2 = chain_map(J2, N::Int64, D::Float64) #empty chain

        g1 = h1[1] #filled chain to system coupling
        g2 = h2[2] #empty chain to system coupling

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
        (;N_L, N_R, β_L, β_R, μ_L, μ_R, D_L, D_R, E_sys) = P

        if N_L == 0 
            H_R, g1_R, g2_R = H_bath(N_R, J_R, D_R, β_R, μ_R) #right bath hamiltonian
            H_sys = [0.0 0.0 ; 0.0 E_sys] #system + ancilla hamiltonian, ancilla -> system
            
            H = cat(H_sys, H_R; dims=(1,2))

            #system-bath couplings
            H[2, 3] = g1_R
            H[3, 2] = g1_R
            H[2, 4] = g2_R
            H[4, 2] = g2_R

            return Matrix(H)
        end

        H_L, g1_L, g2_L = H_bath(N_L, J_L, D_L, β_L, μ_L) #left bath hamiltonian
        H_R, g1_R, g2_R = H_bath(N_R, J_R, D_R, β_R, μ_R) #right bath hamiltonian
        H_sys = [0.0 0.0 ; 0.0 E_sys] #system + ancilla hamiltonian, ancilla -> system
        
        H = cat(reverse(H_L), H_sys, H_R; dims=(1,2)) #H_L reversed to attach chain to system properly

        #system-bath couplings
        H[2N_L + 2, 2N_L] = g1_L
        H[2N_L, 2N_L + 2] = g1_L
        H[2N_L + 2, 2N_L - 1] = g2_L
        H[2N_L - 1, 2N_L + 2] = g2_L

        H[2N_L + 2, 2N_L + 3] = g1_R
        H[2N_L + 3, 2N_L + 2] = g1_R
        H[2N_L + 2, 2N_L + 4] = g2_R
        H[2N_L + 4, 2N_L + 2] = g2_R

        return Matrix(H)
    end

    function prepare_corrs(P)
        """creates initial correlation matrix"""
        (;N_L, N_R) = P

        if N_L == 0
            C_R = zeros(ComplexF64, N_R, N_R)
            for n in 1:N_R
                C_R[n,n] = (n%2)+1.0
            end

            C_emp = complex([0.0 0.0 ; 0.0 0.0]) #empty system-ancilla state for BLP measure
            C_full = complex([0.0 0.0 ; 0.0 1.0]) #full system-ancilla state for BLP measure
            C_CJ = complex([0.5 0.5 ; 0.5 0.5]) #anti-correlated system-ancilla state for CJ isomorphism

            C0_emp = cat(C_emp, C_R; dims=(1,2))
            C0_full = cat(C_full, C_R; dims=(1,2))
            C0_CJ = cat(C_CJ, C_R; dims=(1,2))

            return C0_emp, C0_full, C0_CJ
        end

        C_L = zeros(ComplexF64, N_L, N_L)
        for n in 1:N_L
            C_L[n,n] = (n%2)+1.0
        end
        C_R = zeros(ComplexF64, N_R, N_R)
        for n in 1:N_R
            C_R[n,n] = (n%2)+1.0
        end

        C_emp = complex([0.0 0.0 ; 0.0 0.0]) #empty system-ancilla state for BLP measure
        C_full = complex([0.0 0.0 ; 0.0 1.0]) #full system-ancilla state for BLP measure
        C_CJ = complex([0.5 0.5 ; 0.5 0.5]) #anti-correlated system-ancilla state for CJ isomorphism

        C0_emp = cat(reverse(C_L), C_emp, C_R; dims=(1,2))
        C0_full = cat(reverse(C_L), C_full, C_R; dims=(1,2))
        C0_CJ = cat(reverse(C_L), C_CJ, C_R; dims=(1,2))

        return C0_emp, C0_full, C0_CJ
    end

    function evolve_corrs(C0, H, P)
        (;N_L, N_R, D_L, D_R, dt) = P
        Cs = Vector{Array{ComplexF64}}(undef, 0)
        C0 = Matrix(C0)
        H = Matrix(H)
        
        step_max=2000
        if N_L <= N_R
            if D_L <= D_R
                step_max = round(2*N_L/(dt*D_R))
            else
                step_max = round(2*N_L/(dt*D_L))
            end
        else
            if D_L <= D_R
                step_max = round(2*N_R/(dt*D_R))
            else
                step_max = round(2*N_R/(dt*D_L))
            end
        end

        # Pre-compute the single step propagator
        U_dt = exp(-im * dt * H) 
        U_dt_dag = U_dt'
        
        C_curr = C0
        push!(Cs, C_curr)
        
        step = 1
        ans = false
        
        while ans == false 
            # Update state iteratively
            C_curr = U_dt * C_curr * U_dt_dag
            push!(Cs, C_curr)
            
            if step % 100 == 0
                nSys = [real(M[N+1,N+1]) for M in Cs[end-100:end]]
                min,max = extrema(nSys)
                ans = (max-min < 0.001)
            end
            step += 1
            # Safety break to prevent infinite loops if it doesn't settle
            if step > step_max; break; end 
        end
        return Cs
    end

    function calculate_measures(P)
        (;spec_fun, Γ_L, Γ_R, D_L, D_R, dt) = P
        qA = N_L + 1 ; qS = N_L + 2 #ancilla and system index

        J_L = spectral_function(spec_fun, D_L, Γ_L)
        J_R = spectral_function(spec_fun, D_R, Γ_R)
        H = H_tot(J_L, J_R, P)
        C0_e, C0_f, C0_CJ = prepare_corrs(P)
        Cs_e = evolve_corrs(C0_e, H, dt)
        Cs_f = evolve_corrs(C0_f, H, dt)
        Cs_CJ = evolve_corrs(C0_CJ, H, dt)

        
    end

end

