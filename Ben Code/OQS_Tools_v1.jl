module OQS_Tools_v1

    using ITensors
    using ITensorMPS
    using LinearAlgebra
    using PolyChaos
    using QuadGK
    using Plots

    export create_spectral
    export modify_spectral
    export thermofield_transform
    export chain_map
    export make_H_mpo
    export make_H_matrix
    export prepare_MPS
    export evolve_MPS
    export prepare_correlations
    export evolve_correlations  


    function create_spectral(input::AbstractString, D::Float64, g::Float64) # (bool, amp, mean, sd)
        # window
        inband = x -> (-D <= x <= D)
        
        if input == "flat"
            J = x -> inband(x) ? 1/(2D) : 0.0
        elseif input == "elliptical"
            J = x -> inband(x) ? sqrt(1 - (x/D)^2) : 0.0
        else
            error("spectral function type not recognized")
        end
        # normalization
        norm = quadgk(J, -D, D)[1]           
        Jnorm = x -> g*D/pi * J(x) / norm
        return Jnorm
    end

    function modify_spectral(J::Function, input::AbstractString, params::Tuple)
        """modifies the spectral function by adding a feature such as a band gap or gaussian perturbation"""
        Jmod = J 
        return Jmod
    end

    function thermofield_transform(J, beta::Float64, mu::Float64) #spectral function, inverse temp, chemical potential
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
        ortho_poly = OrthoPoly("bath_op", N, meas; Nquad=2000)   
        chain = coeffs(ortho_poly)                                  
        E = chain[1:N,1] #site energies
        h = sqrt.(chain[1:N,2]) #site hoppings (first term is system hopping)
        return E, h
    end

    function prepare_MPS(N::Int64, sys_occ::Float64)
        """prepare initial MPS state, currently just for fermions"""
        sites = siteinds("Fermion", 2*N+1, conserve_qns=true) #assuming that truncated bath remains approximately closed within simulation time

        states = [(j <= N) ? "Occ" : "Emp" for j=1:2*N+1]

        psi0 = productMPS(sites, states)
        return psi0, sites, states
    end

    function make_H_mpo(E1, E2, h1, h2, sites, Es, N, sys::Int64) #filled/empty site energies, filled/empty hoppings, ITensor sites, system onsite energy, chain length, system site index
        """fermionic nearest neighbour Hamiltonian for thermofield chain mapped OQS"""
        ampo = AutoMPO()
        g1 = h1[1]
        g2 = h2[1]
        t1 = reverse(h1[2:end]) #empty chain NN couplings
        E1 = reverse(E1) #empty chain onsite energies
        t2 = h2[2:end] #filled chain NN couplings
        
        for j in 1:N
            add!(ampo, E1[j], "N", j)
        end
        for j in 1:N-1
            add!(ampo, t1[j], "Cdag", j, "C", j+1)
            add!(ampo, t1[j], "Cdag", j+1, "C", j)
        end
        for j in 1:N
            add!(ampo, E2[j], "N", sys + j)
        end
        for j in 1:N-1
            add!(ampo, t2[j], "Cdag", sys + j, "C", sys + j + 1)
            add!(ampo, t2[j], "Cdag", sys + j + 1, "C", sys + j)
        end
        # system onsite
        add!(ampo, Es, "N", sys)
        add!(ampo, g1, "Cdag", sys, "C", N);         add!(ampo, g1, "Cdag", N, "C", sys)
        add!(ampo, g2, "Cdag", sys, "C", sys + 1);    add!(ampo, g2, "Cdag", sys + 1,  "C", sys)
        return MPO(ampo, sites)
    end

    function make_H_matrix(E1, E2, h1, h2, Es, N::Int64) #filled/empty site energies, filled/empty hoppings, chain length, system site index
        """exact diagonalisation Hamiltonian for chain mapped OQS"""
        E1 = reverse(E1) #empty chain onsite energies
        h1 = reverse(h1) #empty chain NN couplings
        d = Vector{Float64}(undef, 2*N+1)     # diagonal 
        e = Vector{Float64}(undef, 2*N)       # off-diagonal
        d[1:N] .= E1
        d[N+1] = Es
        d[N+2:2N+1] .= E2
        e[1:N] .= h1
        e[N+1:2N] .= h2
        H = SymTridiagonal(d, e)            # Hermitian and tridiagonal
        return H

    end

    function evolve_MPS(psi0::MPS, H::MPO, sys::Int64, dt::Float64, tmax::Float64)
        """Time evolve MPS with Hamiltonian MPO using TDVP"""
        sweeps = Sweeps(2); maxdim!(sweeps, 400, 800); cutoff!(sweeps, 1e-9)
        psi = psi0
        ts = collect(dt:dt:tmax)
        len = length(ts)
        nSys = zeros(len)
        for k in 1:len
            psi = tdvp(H, -im*dt, psi; nsite=2, outputlevel=0,mindim=1, maxdim=100) #time_step=dt, nsweeps=sweeps, order=2)
            nSys[k] = expect(psi, "N")[sys]
            println("timestep $k of $len complete")
        end
        return psi, nSys
    end

    function prepare_correlations(N, sys_occ)                             
        nd0 = sys_occ                 
        
        I_L  = I(N)                
        Z_L  = zeros(N, N)

        # assemble C0
        C0 = zeros(ComplexF64, 2N+1, 2N+1)
        C0[N+1,N+1] = nd0 + 0im
        C0[1:N, 1:N] .= I_L    
        C0[2+N:end, 2+N:end] .= Z_L
        C0 = Hermitian(C0)
        return C0
    end

    function evolve_correlations(C, H, dt, tmax, N::Int64)
        ts = collect(dt:dt:tmax)
        F = eigen(H)
        U = Matrix(F.vectors)
        M = U' * Matrix(C) * U
        display(heatmap(real(M)))
        nSys = zeros(length(ts))
        U = exp(-im * dt .* H )
        for k in 1:length(ts)
            C = U * C * U'
            nSys[k] = real(C[N+1,N+1])
        end
        return C, nSys
    end
end #module