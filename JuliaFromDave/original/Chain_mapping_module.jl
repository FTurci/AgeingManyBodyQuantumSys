module Chain_mapping


    include("HamiltonianBuilding.jl")
    using .HamiltonianBuilding


    using ITensors
    using ITensorMPS
    using LinearAlgebra
    using Observers
    using ProgressBars
    using PolyChaos

    export Base_params
    export dependent_params
    export DP_initialisation
    export initialise_indices
    export mode_operators
    export initial_state_gates
    export initialise_psi
    export initial_correlation_matrix
    export create_H_single
    export thermofield_chain_mapping_H_matrix
    export system_Hamiltonian
    export system_bath_couplings
    export initialise_bath
    export orthopol_chain
    export spectral_function
    export thermofield_renormalisation
    export propagate_correlations
    export propagate_MPS
    export boundary_test
    export initialise_observer
    export current_time
    export measure_correlation_matrix
    export measure_SvN
    export measure_spin_dn_electron_correlation_matrix
    export measure_spin_up_electron_correlation_matrix
    export measure_SIAM_diag_elements_spinful
    export entanglement_entropy
    export enrich_state!
    export Krylov_states
    export bipart_maxdim
    export enrich

    mutable struct Base_params
    ###Bath params
    N_L ::Int64                      #Number of left bath sites
    N_R ::Int64                      #Number of right bath sites
    Ns  ::Int64                      #Number of system sites (not including ancilla)  
    N_chain ::Int64                  #Number of sites used for the chain mapping, after which it's assumed the coefficients are constant
    β_L ::Float64                    #inverse temperature of right bath
    β_R ::Float64                    #inverse temperature of left bath
    μ_L ::Float64                   #chemical potential of left bath
    μ_R ::Float64                   #chemical potential of right bath
    Γ_L ::Float64                    #Left coupling strength
    Γ_R ::Float64                    #Right coupling strength
    D ::Float64                      #Half the bandwidth 
    spec_fun_type ::String           #Spectral function choice
    symmetry_subspace ::String       #Choice of whether to only consider modes in a given symmetry subspace of the Hilbert space
    bath_mode_type ::String          #Type of bath mode e.g. Fermions, Bosons, spins etc (this code is only valid for Fermion and Electron sitetypes)
    sys_mode_type ::String           #Type of system mode e.g. Fermions, Bosons, spins etc (this code is only valid for Fermion and Electron sitetypes)
    ϵ ::Float64                      #Energy of system modes
    tc ::Float64                     #System hopping
    init_occ ::Float64               #Initial occupation of system modes
    U ::Float64                      #System interaction strength
    δt ::Float64                     #timestep
    T ::Float64                      #Evolution time
    Kr_cutoff ::Float64              #Numerical cutoff used for Krylov enrichment
    k1 ::Int64                       #Number of Krylov states used
    τ_Krylov ::Float64
    n_enrich ::Int64                 #Number of timesteps between each enrichment
    tdvp_cutoff ::Float64            #Numerical cutoff for tdvp
    minbonddim ::Int64               #Minimum bond dimension for tdvp 
    maxbonddim ::Int64               #Maximum bond dimension for tdvp
    T_enrich ::Float64               #Time in simulation where enrichment is used
    Base_params() = new()
end

    mutable struct dependent_params
        """
        Struct holding dependent parameters defined by Base_params.
        """

        N ::Int64                        #number of modes including ancillas
        ϵi ::Vector{Float64}             #self energies of system modes
        ti ::Vector{Float64}             #coupling of system modes
        Ui ::Vector{Float64}             #system interaction strength
        qB_L ::UnitRange{Int64}          #Indices for left bath
        qB_R ::UnitRange{Int64}          #Indices for right bath
        qS ::StepRange{Int64, Int64}     #Indices for system
        qtot ::UnitRange{Int64}          #indices for full MPS
        times ::Vector{Float64}          #Evolution time array
        bath_ann_op ::Any                #Bath annihilation operator
        bath_cre_op ::Any                #Bath creation operator
        sys_ann_op ::Any                 #system annihilation operator
        sys_cre_op ::Any                 #system creation operator
        s ::Any                          #ITensor indices
        c ::Any                          #ITensor annihilation operators
        cdag ::Any                       #ITensor annihilation operators
        Id ::Any                         #Identities
        F ::Any                          #Pauli Z operators
        Ci ::Matrix{ComplexF64}          #Initial correlation matrix
        H_single ::Matrix{ComplexF64}    #Single particle hamiltonian
        H_MPO ::MPO                      #Hamiltonian matrix product operator
        MPO_terms ::Any                  #MPO terms for H_MPO
        ψ_init ::MPS                     #Initial state as an MPS
        T_unenriched ::Float64           #T-T_enrich, total time evolution when enrichment is not used
        dependent_params() = new()
    end

    function DP_initialisation(P;kwargs...)    

        """
        kwargs: ϵi,ti,init_occ_vec
        """

        (;Ns,N_L,N_R,δt,T,ϵ,tc,U,sys_mode_type,bath_mode_type) = P
        DP = dependent_params()

        #Initialise the index objects needed for other functions
        initialise_indices(P,DP)

        #times array
        DP.times = range(δt,stop=T,step =δt)       

        ##Defining creation and annihilation operators depending on whether the sites are spinless or not. They're set up 
        ##as arrays with the indices representing the spin, which means it's a bit clunky for the spinless case as it's only 1 index.
        ##Note that I have kept the distinction between the system mode type and the bath mode type, but this code assumes they're the same for simplicity.

        if bath_mode_type == "Fermion"
            DP.bath_ann_op = ["C"]
            DP.bath_cre_op = ["Cdag"]
        elseif bath_mode_type == "Electron"
            DP.bath_ann_op = ["Cdn","Cup"]
            DP.bath_cre_op = ["Cdagdn","Cdagup"]
        end
        if sys_mode_type == "Fermion"
            DP.sys_ann_op = ["C"]
            DP.sys_cre_op = ["Cdag"]
        elseif bath_mode_type == "Electron"
            DP.sys_ann_op = ["Cdn","Cup"]
            DP.sys_cre_op = ["Cdagdn","Cdagup"]
        end

        # Array of site indices
        DP.s,DP.cdag,DP.c = mode_operators(DP,P)                         

        #self energies of system modes
        DP.ϵi = get(kwargs,:ϵi,ϵ*ones(Ns));   
        #coupling of system modes                          
        DP.ti = get(kwargs,:ti,tc*ones(Ns-1));    

        #Interactions between modes. for spinless fermions this is a
        #Ns-1 vector as the interaction term is between sites, whereas for the spinful case
        #it's an onsite term between spin so has length Ns
        if sys_mode_type == "Electron"
            DP.Ui = U*ones(Ns);
        else 
            DP.Ui = U*ones(Ns-1)
        end    

        #Pauli Z operators needed for JW strings (Not actually used in this code)
        DP.F = ops(DP.s, [("F", n) for n in DP.qtot]) 

        #List of MPS identities
        DP.Id = Vector{ITensor}(undef,length(DP.s))   
        for i =1:length(DP.s)
            iv = DP.s[i]
            ID = ITensor(iv', dag(iv));
            for j in 1:ITensors.dim(iv)
                ID[iv' => j, iv => j] = 1.0
            end
            DP.Id[i] = ID
        end

        DP.ψ_init = initialise_psi(P,DP;kwargs...)
        DP.Ci = initial_correlation_matrix(DP)

        DP.H_single = create_H_single(P,DP)
        HS_os,DP.H_single = system_Hamiltonian(DP.H_single,P,DP)
        couplings,DP.H_single =  system_bath_couplings(DP.H_single,P,DP)
        DP.MPO_terms = HS_os + couplings
        if N_L >0
            [DP.MPO_terms += build_from_matrix(DP.H_single[DP.qB_L,DP.qB_L],DP.bath_cre_op[i],DP.bath_ann_op[i]) for i in 1:length(DP.bath_ann_op)]
        end

        if N_R >0
            offset = 2*N_L+Ns
            [DP.MPO_terms += build_from_matrix(DP.H_single[DP.qB_R,DP.qB_R],DP.bath_cre_op[i],DP.bath_ann_op[i];offset= offset) for i in 1:length(DP.bath_ann_op)]
        end

        DP.H_MPO = MPO(DP.MPO_terms,DP.s)     
        DP.T_unenriched = round(P.T-P.T_enrich,digits=10)       # Time when the state is no longer enriched each step                                       
        return DP
    end



    function initialise_indices(P,DP)

        (;N_L,N_R,N_chain,Ns) = P
        
        N_R >0 && @assert(N_chain <= N_R)
        N_L >0 && @assert(N_chain <= P.N_L)


        ##Total number of modes in the MPS. The factor of 2 for N_L and N_R
        ##is due to the thermofield purification
        DP.N = 2*N_L+2*N_R +Ns
        
        ##Indices for left bath
        DP.qB_L = N_L == 0 ? (0:0) : 1:2*N_L

        #right bath starts at 2*N_L+Ns+1
        DP.qB_R = N_R == 0 ? (0:0) : (DP.qB_L[end] + Ns + 1):DP.N

        ##indices for system 
        DP.qS = DP.qB_L[end] + 1:(DP.qB_L[end] + Ns)

        ##indices for total MPS
        DP.qtot = 1:DP.N 
    end

    function mode_operators(DP,P)
        (;Ns,N_L,N_R,sys_mode_type,bath_mode_type,symmetry_subspace) = P
        (;qS,qB_L,qB_R,bath_ann_op,bath_cre_op,sys_ann_op,sys_cre_op) = DP  

        """
        This is a fairly clunky function with repeating code, I'm sure it can be made more neat. 
        """

        left_bath_bool = N_L>0
        right_bath_bool = N_R>0

        ##Symmetry subspace choice
        if symmetry_subspace == "Number conserving"
            sys_modes = siteinds(sys_mode_type,Ns;conserve_nf=true)
        elseif symmetry_subspace == "Number and Sz conserving"
            sys_modes = siteinds(sys_mode_type,Ns;conserve_qns=true)
        else
            sys_modes = siteinds(sys_mode_type,Ns)
        end
        
        ##Setting the labels for the system modes
        sys_modes = [settags(mode,sys_mode_type*",System Site,n="*string(i+qB_L[end])) for (i,mode) in enumerate(sys_modes)]
        
        ##s is the set of indices for all the modes, but I create it by first assigning it as just the system modes, then adding the
        ##bath modes.
        s = sys_modes

        ##Defining the mode indices for the left bath
        if left_bath_bool
            if symmetry_subspace == "Number conserving"
                NL_modes = siteinds(bath_mode_type,length(qB_L);conserve_nf=true)
            elseif symmetry_subspace == "Number and Sz conserving"
                NL_modes = siteinds(bath_mode_type,length(qB_L);conserve_qns=true)
            else
                NL_modes = siteinds(bath_mode_type,length(qB_L))
            end
            NL_modes =[settags(mode,bath_mode_type*",Left bath site,n="*string(i)) for (i,mode) in enumerate(NL_modes)]
            s = append!(NL_modes,s)
        end

        ##Defining the mode indices for the right bath
        if right_bath_bool 
            if symmetry_subspace == "Number conserving"
                NR_modes = siteinds(bath_mode_type,length(qB_R);conserve_nf=true)
            elseif symmetry_subspace == "Number and Sz conserving"
                NR_modes = siteinds(bath_mode_type,length(qB_R);conserve_qns=true)
            else
                NR_modes = siteinds(bath_mode_type,length(qB_R))
            end
            NR_modes = [settags(mode,bath_mode_type*",Right bath site,n="*string(i+qS[end])) for (i,mode) in enumerate(NR_modes)]
            s = append!(s,NR_modes)
        end

        ##Set up in the same way as s, but first defining the creation and annihilation operators as the system creation and annihilation operators,
        ##where the index i loops over spin.

        ann_ops = [ops(s, [(sys_ann_op[i], n) for i in 1:length(sys_ann_op)]) for n in qS]
        cre_ops = [ops(s, [(sys_cre_op[i], n) for i in 1:length(sys_ann_op)]) for n in qS]
        
        if left_bath_bool
            ann_NL_modes = [ops(s, [(bath_ann_op[i], n) for i in 1:length(bath_ann_op)]) for n in qB_L]
            cre_NL_modes = [ops(s, [(bath_cre_op[i], n) for i in 1:length(bath_ann_op)]) for n in qB_L]
            ann_ops = append!(ann_NL_modes,ann_ops)
            cre_ops = append!(cre_NL_modes,cre_ops)
        end 

        if right_bath_bool
            ann_NR_modes = [ops(s, [(bath_ann_op[i], n) for i in 1:length(bath_ann_op)]) for n in qB_R]
            cre_NR_modes = [ops(s, [(bath_cre_op[i], n) for i in 1:length(bath_ann_op)]) for n in qB_R]
            ann_ops = append!(ann_ops,ann_NR_modes)
            cre_ops = append!(cre_ops,cre_NR_modes)
        end

        return s,cre_ops,ann_ops
    end

    function initial_state_gates(P,DP;kwargs...)

        (;Ns,sys_mode_type,init_occ) = P
        (;cdag,Id,s,qS) = DP


        #defaults to a uniform occupation, can override this with the kwarg init_occ_vec
        init_occ_vec = get(kwargs,:init_occ_vec,init_occ*ones(length(qS)))
        
        ##This step is to remove the Z factor on the spin up space of the site for cdagdn. we are free to do this
        ## as it is just choosing an ordering {cdagup_1,cdagdn_1,cdagup_2,cdagdn_2,...,cdagup_N,cdagdn_N} We apply the 
        ##gates from the right so the JW strings always act on the vacuum so don't have any impact.
        if sys_mode_type == "Electron"
            cdag = [ops(s, [(cre_op, n) for cre_op in ["Adagdn","Adagup"]]) for n in 1:length(s)]
        end

        init_occ_vec = get(kwargs,:init_occ_vec,init_occ*ones(length(qS)))

        ##Can't remember why I separated into these three
        if init_occ_vec == zeros(length(qS))
            system_gate = [Id[n]  for n in qS]
        elseif init_occ_vec == ones(length(qS))
            system_gate = [cdag[n][i] for i in 1:length(cdag[1]) for n in qS]
        else
            system_gate = [(√(init_occ_vec[j])*cdag[n][i] + √(1-init_occ_vec[j])*Id[n]) for i in 1:length(cdag[1])  for (j,n) in enumerate(qS)]
        end

        return system_gate
    end

    function initialise_psi(P,DP;kwargs...)
        (;Ns,N_L,N_R,bath_mode_type) = P
        (;s,qtot,qB_L,qB_R,N) = DP    
        
        """
        NOTE: This function assumes an interleaved ordering.
        """

        if bath_mode_type == "Electron"
            Empty = "Emp"
            Full = "UpDn"
        else
            Empty = "0"
            Full = "1"
        end
        #Initialiseing the occupations as an empty set
        occs = [Empty for n in qtot]

        ##Creating the thermofield occupations in interleaved ordering
        bath_occs = [Full,Empty]
        N_L>0 && (occs[qB_L] = repeat(bath_occs,Int(length(qB_L)/2)))
        N_R>0 && (occs[qB_R] = repeat(bath_occs,Int(length(qB_R)/2)))

        ##Defining an MPS with the thermal occupations
        therm = MPS(ComplexF64,s,occs)

        ##creates the system initial state. These don't include JW strings on the left bath as
        ## it's in a fock state so will only give an overall factor of -1 or 1. Can also justify this by thinking
        ##about exciting the whole MPS from right to left so JW strings always act on the vacuum.
        system_gate = initial_state_gates(P,DP;kwargs...)
        ψ_init = apply(system_gate,therm;cutoff=1e-15)
        
        # Normalize state just in case
        orthogonalize!(ψ_init,N)
        ψ_init[DP.N]= ψ_init[N]/norm(ψ_init)
        @show(norm(ψ_init))
        
        return ψ_init

    end

    function initial_correlation_matrix(DP)

        (;qtot,bath_cre_op,bath_ann_op,ψ_init,N) = DP
        
        ##initialising the correlation matrix
        C = complex(zeros(length(bath_ann_op)*N,length(bath_ann_op)*N))
        
        ##This loop is basically just for the spinful case, where I arrange C such that
        ##C[1:N,1:N] is the correlation matrix for the down spins, and C[N+1:2*N:N+1:2*N]
        ##is the correlation matrix for the up spins.
        for (i, cre) in enumerate(bath_cre_op), (j, ann) in enumerate(bath_ann_op)
            C[qtot .+ (i-1)*N, qtot .+ (j-1)*N] = transpose(correlation_matrix(ψ_init, cre, ann))
        end
        return C
    end

    function create_H_single(P,DP;kwargs...)
        (;N_L,N_R) = P
        (;N) = DP

        ##This doesn't include the system terms or system bath coupling terms.

        H_single = complex(zeros(N,N))
        if N_L>0
            H_single = thermofield_chain_mapping_H_matrix(H_single,"left",P,DP)
        end
        if N_R>0
            H_single = thermofield_chain_mapping_H_matrix(H_single,"right",P,DP)
        end
        return H_single
    end

    function thermofield_chain_mapping_H_matrix(H_single,side,P,DP)
        (;N_L,N_R,Ns,bath_mode_type) = P
        (;N) = DP

        ##This deals with the Hamiltonian of the thermofield baths in isolation, the system
        ##bath couplings are dealt with in a separate function
        if side == "left" 
            Vk_emp_L,ϵb_emp_L,Vk_fill_L,ϵb_fill_L = initialise_bath("left",P)  
            left_bath_mode_inds = 1:2:2*(N_L)
            b = 0
            for j in left_bath_mode_inds
                b += 1
                H_single[j,j] = ϵb_fill_L[b]
                H_single[j+1,j+1] =  ϵb_emp_L[b]
                if j<(2*N_L-1)
                    H_single[j,j+2] = Vk_fill_L[b]
                    H_single[j+2,j] = conj(Vk_fill_L[b])
                    H_single[j+1,j+3] =  Vk_emp_L[b]
                    H_single[j+3,j+1] =  conj(Vk_emp_L[b])
                end
            end
        end
        if side == "right"
            Vk_emp_R,ϵb_emp_R,Vk_fill_R,ϵb_fill_R = initialise_bath("right",P)
            right_bath_mode_inds = 2*N_L+Ns+1:2:N
            b = 0
            for j in right_bath_mode_inds
                b += 1   
                H_single[j,j] = ϵb_fill_R[b]
                H_single[j+1,j+1] = ϵb_emp_R[b]
                if j>right_bath_mode_inds[1]
                    H_single[j-2,j] = conj(Vk_fill_R[b])
                    H_single[j,j-2] = Vk_fill_R[b]
                    H_single[j-1,j+1] =  conj(Vk_emp_R[b])
                    H_single[j+1,j-1] =  Vk_emp_R[b]
                end
            end
        end
        return H_single
    end

    function system_Hamiltonian(H_single,P,DP)
        (;sys_mode_type,Ns,N_L) = P
        (;ϵi,ti,Ui,qS,s) = DP     
        os = OpSum()
        


        for i = 1:Ns
            if sys_mode_type == "Fermion"
                os += ϵi[i],"n",qS[i]
                H_single[qS[i],qS[i]] = ϵi[i]
                if i<Ns
                    os += ti[i],"Cdag",qS[i],"C",qS[i+1]
                    os += conj(ti[i]),"Cdag",qS[i+1],"C",qS[i]
                    H_single[qS[i],qS[i+1]] = ti[i]
                    H_single[qS[i+1],qS[i]] = conj(ti[i])
                    os += Ui[i],"n",qS[i],"n",qS[i+1]
                end
            elseif sys_mode_type == "Electron"
                os += ϵi[i],"Ntot",qS[i]
                H_single[qS[i],qS[i]] = ϵi[i]
                os += Ui[i],"Nup",qS[i],"Ndn",qS[i]
                if i<Ns
                    for spin in ["up", "dn"]
                        os += ti[i], "Cdag$spin", qS[i], "C$spin", qS[i+1]
                        os += conj(ti[i]), "Cdag$spin", qS[i+1], "C$spin", qS[i]
                    end
                    H_single[qS[i],qS[i+1]] = ti[i]
                    H_single[qS[i+1],qS[i]] = conj(ti[i])
                end
            end
        end
        return os,H_single
    end

    function system_bath_couplings(H_single,P,DP)
        (;N_L,N_R,Ns) = P
        (;qB_L,qB_R,qS,bath_cre_op,bath_ann_op,sys_ann_op,sys_cre_op) = DP

        os = OpSum()
        
        ##Not always the case that the coupling operator is just the creation/annihilation operators.
        coupling_ann_op = sys_ann_op
        coupling_cre_op = sys_cre_op

        if N_L>0
            Vk_emp_L,ϵb_emp_L,Vk_fill_L,ϵb_fill_L = initialise_bath("left",P)
            
            """Thermofield left empty bath to system"""
            H_single[2*N_L,qS[1]] =  Vk_emp_L[end]
            H_single[qS[1],2*N_L] =  conj(Vk_emp_L[end])

            """Thermofield left full (only for fermions, both are empty for bosons) bath to system"""
            H_single[2*N_L-1,qS[1]] =  sym_factor*Vk_fill_L[end]
            H_single[qS[1],2*N_L-1] =  sym_factor*conj(Vk_fill_L[end])

            ##Loops over spin
            for i in 1:length(bath_cre_op)
                #Empty bath
                os += Vk_emp_L[end],bath_cre_op[i],qB_L[end],coupling_ann_op[i],qS[1]
                os += conj(Vk_emp_L[end]),coupling_cre_op[i],qS[1],bath_ann_op[i],qB_L[end]

                #Filled bath
                os += sym_factor*Vk_fill_L[end],bath_cre_op[i],qB_L[end]-1,coupling_ann_op[i],qS[1]
                os += sym_factor*conj(Vk_fill_L[end]),coupling_cre_op[i],qS[1],bath_ann_op[i],qB_L[end]-1
            end
        end

        if N_R>0
            Vk_emp_R,ϵb_emp_R,Vk_fill_R,ϵb_fill_R = initialise_bath("right",P)

            """Thermofield right empty bath to system"""
            H_single[qB_R[2],qS[end]] = Vk_emp_R[1]
            H_single[qS[end],qB_R[2]] = conj(Vk_emp_R[1])

            """Thermofield right full (only for fermions, both are empty for bosons) bath to system"""
            H_single[qB_R[1],qS[end]] =  Vk_fill_R[1]
            H_single[qS[end],qB_R[1]] =  conj(Vk_fill_R[1])

                ##Loops over spin
            for i in 1:length(bath_cre_op)
                #Empty bath
                os += Vk_emp_R[1],bath_cre_op[i],qB_R[2],coupling_ann_op[i],qS[end] 
                os += conj(Vk_emp_R[1]),coupling_cre_op[i],qS[end],bath_ann_op[i],qB_R[2] 

                #Filled bath
                os += Vk_fill_R[1],bath_cre_op[i],qB_R[1],coupling_ann_op[i],qS[end]
                os += conj(Vk_fill_R[1]),coupling_cre_op[i],qS[end],bath_ann_op[i],qB_R[1]
            end
        end
        return os,H_single
    end

    function initialise_bath(side,P)
        Vk_emp, ϵb_emp = orthopol_chain(1,side,P)
        Vk_fill, ϵb_fill = orthopol_chain(2,side,P)
        
        if side =="left"
            Vk_emp = reverse(Vk_emp)
            Vk_fill = reverse(Vk_fill)
            ϵb_emp = reverse(ϵb_emp)
            ϵb_fill = reverse(ϵb_fill)
            
        end
        return Vk_emp,ϵb_emp,Vk_fill,ϵb_fill
    end

    function orthopol_chain(thermo_chain_number,side,P)
        """
        Implements chain mapping using orthogonal polynomials.
        thermo_chain_number:
        - 0 full spectral density, no thermofield
        - 1 (1-occupation_number)*spectral density (empty chain)
        -2 occupation_number*spectral density (filled chain)
        """
        (;D,bath_mode_type,N_L,N_R,N_chain) = P
        
        if side=="left"
            Nb = N_L
            β = P.β_L
        elseif side=="right"
            Nb = N_R
            β = P.β_R
        else
            error("No side chosen.")
        end
        couplings,energies = complex(zeros(Nb)),complex(zeros(Nb))
        
        
        w(t) = spectral_function(t, thermo_chain_number, side, P)
        ##Needs to be bigger than the true support (can't really remember why)
        supp = (-2D,2D)

        #degree = Nb-1
        my_meas = Measure("my_meas", w, supp, false, Dict())
        my_op = OrthoPoly("my_op", N_chain-1, my_meas; Nquad=100000);
        α_coeffs,β_coeffs = coeffs(my_op)[:,1],coeffs(my_op)[:,2]

        if length(α_coeffs)<Nb
            energies[1:N_chain] = α_coeffs
            couplings[1:N_chain] = sqrt.(β_coeffs)
            energies[N_chain+1:end] .= α_coeffs[N_chain]
            couplings[N_chain+1:end] .= sqrt(β_coeffs[N_chain])
        else
            energies = α_coeffs
            couplings = sqrt.(β_coeffs)
        end

        return couplings,energies
    end

    function spectral_function(w,thermo_chain_number,side,P;kwargs...)
        """
        This function creates the spectral function for either the left or right bath 
        depending on "side". thermo_chain_number denotes whether the bath is a filled chain, empty chain 
        or the full chain with no thermofield transform.
        """
        (;bath_mode_type) = P

        ###Choosing which bath
        if side =="left"
            spec_fun_type,Γ,β,μ,D= P.spec_fun_type,P.Γ_L,P.β_L,P.μ_L,P.D
        elseif side == "right"
            spec_fun_type,Γ,β,μ,D = P.spec_fun_type,P.Γ_R,P.β_R,P.μ_R,P.D
        else 
            error("neither side chosen")
        end
        renorm = thermofield_renormalisation(w,β,μ,thermo_chain_number)
        
        ##Choosing which spectral function to use
        if spec_fun_type == "box"
            ρ = (1/(2*D))*(heaviside(w .+ D) .- heaviside(w .- D)).*renorm
            J =  (Γ*D/π)*ρ
        elseif spec_fun_type =="ellipse"
            ρ = real((2/(π*D))*sqrt.(Complex.(1 .-(w/D).^2)).*renorm)
            J =  (Γ*D/π)*ρ
            

        elseif spec_fun_type =="Lorentzian"
            λ = get(kwargs,:λ,0.2)
            α = 1/(2*λ*atan(D/λ))
            ρ = 1 .+(w./λ).^2
            ρ = (α./ρ).*renorm
            J =  (Γ*D/π)*ρ
        elseif spec_fun_type == "symmetric ohmic"
            ρ = (w.^2).*exp.(-abs.(w) ./ωc)
            factor = 2*2*ωc^3-ωc*(2*ωc^2+2*ωc*D+D^2)*exp(-D/ωc)
            ρ = ρ/factor
            ρ = ρ.*renorm
            J =  (Γ*D/π)*ρ
        elseif spec_fun_type == "smoothed box"

            """
            Based on Influence functional paper by Abanin: An efficient method for quantum impurity problems out of equilibrium
            """

            ν =  100
            ωc = 1
            denominator = (1 .+exp.(ν*(w .-ωc))).*(1 .+exp.(-ν*(w .+ωc)))
            J = (Γ/(2*π)) ./denominator
            J = J.*renorm
            error("No spectral density chosen")
        end
        return J
    end

    function thermofield_renormalisation(w,β,μ,thermo_chain_number)

        fw = f = 1 ./(exp.(β*(w .- μ)).+ 1)
        
        renorm = 1
        if thermo_chain_number == 0
            renorm = 1
        elseif thermo_chain_number == 1
            renorm = 1 .-fw
        elseif thermo_chain_number == 2 
            renorm = fw
        else 
            error("no chain chosen for thermofield renormalisation")
        end
        return renorm
    end

    """
    Time evolution functions
    """

    function propagate_correlations(P,DP)
        """
        The correlation matrix C_ij = expect(cdag[j]*c[i]) propagates according to
        C_ij(t) =U*C_ij(0)*U', but G_ij = expect(cdag[i]*c[j]) doesn't.
        """
        (;sys_mode_type,bath_mode_type) = P
        (;Ci, H_single,times,N) = DP
        δt = times[2] - times[1]
        
        if sys_mode_type == "Electron"
            @assert(sys_mode_type == bath_mode_type)
            H_single = [H_single zeros(N,N);zeros(N,N) H_single]
        end
        U_step = exp(-im*δt*H_single)

        corrs = Vector{Any}(undef,length(times))
        corrs[1] = U_step*Ci*U_step'
        for i in 2:length(times)
            corrs[i] = U_step*corrs[i-1]*U_step'
        end
        return corrs
    end

    function propagate_MPS(P, DP; kwargs...)
        # Parameter unpacking
        (; n_enrich, δt, tdvp_cutoff, minbonddim, maxbonddim,T_enrich) = P
        (; H_MPO, s) = DP

        obs = initialise_observer(P, DP, get(kwargs, :obs, false))
        enrich_bool = get(kwargs,:enrich_bool,true)
        TDVP_nsite = get(kwargs,:TDVP_nsite,2)
        boundary_test_bool = get(kwargs,:boundary_test_bool,true)
        times = get(kwargs,:times,DP.times)
        ψ = get(kwargs,:ψ_init,deepcopy(DP.ψ_init))    
        
        # Configure updater parameters
        updater_kwargs = Dict(:ishermitian => true, :issymmetric => true, :eager => true) # FT: WE WANT THIS LINE
        normalize = true

        #Normalisation
        ψ = ψ/norm(ψ)
        
        ##Enrichment
        TDVP_nsite == 2 && enrich_bool && (ψ = enrich_state!(ψ, P, DP))

        global sim_t = 0

        # Main time evolution loop
        for i in 1:length(times)
            @time ψ = ITensorMPS.tdvp(H_MPO, -im * δt, ψ; time_step = -im * δt, cutoff = tdvp_cutoff, 
                                    mindim = minbonddim, maxdim = maxbonddim, outputlevel = 1, 
                                    normalize = normalize, observer! = obs, updater_kwargs, 
                                    nsite = TDVP_nsite, reverse_step = true)
            global sim_t += δt
            @show(sim_t)

            if (i % n_enrich == 0)
                # Enrichment logic
                if (sim_t <= T_enrich) && TDVP_nsite == 2 && enrich_bool
                    println("Time taken for enrichment")
                    @time ψ = enrich_state!(ψ, P, DP)
                end
            end
            # Boundary condition check
            if boundary_test_bool && sum(boundary_test(ψ,1e-3, DP, P)) > 0
                println("Boundary reached at t = $sim_t")
                break
            end
        end
        return ψ, obs
    end

    function boundary_test(ψ,tol,DP,P)
        (;N_L,N_R,bath_mode_type) = P
        (;N,ψ_init,qB_L,qB_R,sys_cre_op,sys_ann_op) = DP
        left_bath_bool = N_L>0
        right_bath_bool = N_R>0
        
        if bath_mode_type == "Electron"
            density_op = "Nup"
        else
            density_op = "n"
        end
        if left_bath_bool
            left_boundary_test = expect(ψ_init,density_op;sites=qB_L)[1] - expect(ψ,density_op,sites=qB_L)[1]
            left_bool = abs(left_boundary_test)>tol
            if left_bool
                @show(left_boundary_test)
            end
        else 
            left_bool = false
        end
        if right_bath_bool
            right_boundary_test = expect(ψ_init,density_op;sites=qB_R)[end]- expect(ψ,density_op;sites=qB_R)[end]
            right_bool = abs(right_boundary_test)>tol
            if right_bool
                @show(right_boundary_test)
            end
        else
            right_bool = false
        end
        return left_bool,right_bool
    end



    """
    Measurement functions for tdvp
    """

    function initialise_observer(P, DP, obs)
        if obs != false
            return obs
        else
            obs_map = Dict(
                "TFCM_Fermion" => Observer("times" => current_time, "corr" => measure_correlation_matrix, 
                                                "SvN" => measure_SvN),
                "TFCM_Electron" => Observer("times" => current_time, "corr_dn" => measure_spin_dn_electron_correlation_matrix,
                                                "corr_up" => measure_spin_up_electron_correlation_matrix,"SvN" => measure_SvN,
                                                "diag_elements" => measure_SIAM_diag_elements_spinful),
            )
            key = join(["TFCM", P.sys_mode_type], "_")

            return get(obs_map,key,Observer())
        end
    end

    function current_time(; current_time, bond, half_sweep)
        if bond == 1 && half_sweep == 2
        return real(im*current_time)
        end
        return nothing
    end

    function measure_correlation_matrix(; state, bond, half_sweep)
        if bond==1 && half_sweep == 2
            return transpose(correlation_matrix(state,"Cdag","C"))
        end
        return nothing
    end

    function measure_SvN(; state, bond, half_sweep)
        if bond == 1 && half_sweep == 2
            return entanglement_entropy(state)
        end
        return nothing
    end

    function measure_spin_up_electron_correlation_matrix(; state, bond, half_sweep)
        if bond==1 && half_sweep == 2
            return transpose(correlation_matrix(state,"Cdagup","Cup"))
        end
        return nothing
    end 

    function measure_spin_dn_electron_correlation_matrix(; state, bond, half_sweep)
        if bond==1 && half_sweep == 2
            return transpose(correlation_matrix(state,"Cdagdn","Cdn"))
        end
        return nothing
    end

    function measure_SIAM_diag_elements_spinful(; state, bond, half_sweep)
        if bond == 1 && half_sweep == 2
            s = siteinds(state)
            q = []
            for (i,ind) in enumerate(s)
                if hastags(ind,"SystemSite") == true
                    push!(q,i)
                end
            end
            spin_up_op = op("Nup",s[q[1]])
            spin_dn_op = op("Ndn",s[q[1]])
            Id = op("Id",s[q[1]])

            projector_11 = apply(spin_up_op,spin_dn_op)
            projector_00 = apply((Id -spin_up_op),(Id -spin_dn_op))
            projector_10 = apply(spin_dn_op,(Id  - spin_up_op))
            projector_01 =  apply(spin_up_op,(Id  - spin_dn_op))
            diag_elements = zeros(4)
            diag_elements[1] = real(inner(state,apply(projector_00,state)))
            diag_elements[2] = real(inner(state,apply(projector_01,state)))
            diag_elements[3] = real(inner(state,apply(projector_10,state)))
            diag_elements[4] = real(inner(state,apply(projector_11,state)))


            return diag_elements
        end
        return nothing
    end;


    function entanglement_entropy(ψ)
        # Compute the von Neumann entanglement entropy across each bond of the MPS
        N = length(ψ)
        SvN = zeros(N)
        psi = ψ
        for b=1:N
            psi = orthogonalize(psi, b)
            if b==1
                U,S,V = svd(psi[b] , siteind(psi, b))
            else
                U,S,V = svd(psi[b], (linkind(psi, b-1), siteind(psi, b)))
            end
            for n=1:ITensors.dim(S, 1)
                p = S[n,n]^2
                SvN[b] -= p * log2(p)
            end
        end
        return SvN
    end

    """
    Enrichment functions, based on the algorithm in https://arxiv.org/abs/2005.06104
    """


    function enrich_state!(ψ, P, DP;kwargs...)
        (;s) = DP
        normalise = get(kwargs,:normalise,true)

        Krylov = Krylov_states(ψ, P, DP)
        ψ2 = enrich(ψ, Krylov; P,normalise=normalise)
        if normalise
            ψ[DP.N] = ψ[DP.N] / norm(ψ)
        end

        @show(1 - inner(ψ2, ψ))
        ψ = ψ2
        return ψ
    end

    function Krylov_states(ψ,P,DP;kwargs...)
        (;k1,τ_Krylov) = P
        (;s,H_MPO) = DP
        ishermitian = get(kwargs, :ishermitian, true)
    
        ##Create the first k Krylov states
        Id = MPO(s,"Id")
        Kry_op = 1
        try
            Kry_op = Id-im*τ_Krylov*H_MPO
        catch 
            Kry_op = H_MPO
        end

        list = []
        term = copy(ψ)
    
        for i =1:k1-1
            term = noprime(Kry_op*term)
            term = term/norm(term)
            push!(list,term)
        end
    
        return list
    end

    function bipart_maxdim(s,n)

        left_maxdim = 2^n
        right_maxdim = 2^(length(s)-n)
        if left_maxdim==0
            left_maxdim = 2^63
        end
        if right_maxdim==0
            right_maxdim = 2^63
        end
        return min(left_maxdim,right_maxdim)
    end;

    function enrich(ϕ, ψ⃗; P, kwargs...)
        (;Kr_cutoff) = P
        """
            Given spec from the eigen function, to extract its information use the 
            following functions:

            eigs(spec) returns the spectrum
            truncerror(spec) returns the truncation error
        """  

        normalise = get(kwargs,:normalise,true)

        Nₘₚₛ = length(ψ⃗) ##number of MPS

        @assert all(ψᵢ -> length(ψ⃗[1]) == length(ψᵢ), ψ⃗) ##check that all MPS inputs are of the same length

        N = length(ψ⃗[1]) 
        ψ⃗ = copy.(ψ⃗)

        ###Isn't this already a vector of MPS's?  
        ψ⃗ = convert.(MPS, ψ⃗)

        s = siteinds(ψ⃗[1])
        ##makes the orthogonality centre for each MPS to be at site N  
        ψ⃗ = orthogonalize.(ψ⃗, N)
        ϕ = orthogonalize!(ϕ, N)

        ##storage MPS
        phi = deepcopy(ϕ)

        ρϕ = prime(ϕ[N], s[N]) * dag(ϕ[N])
        ρ⃗ₙ = [prime(ψᵢ[N], s[N]) * dag(ψᵢ[N]) for ψᵢ in ψ⃗]
        ρₙ = sum(ρ⃗ₙ)

        """
        Is this needed?
        """
        ρₙ /=tr(ρₙ)

     # Maximum theoretical link dimensions

        Cϕprev = ϕ[N]
        C⃗ₙ = last.(ψ⃗)


        for n in reverse(2:N)
            """
        In the paper they propose to do this step with no truncation. At the very
        least this cutoff should be a function parameter.
        """    

        left_inds = linkind(ϕ,n-1)

                #Diagonalize primary state ψ's density matrix    
        U,S,Vϕ,spec = svd(Cϕprev,left_inds; 
            lefttags = tags(linkind(ϕ, n - 1)),
            righttags = tags(linkind(ϕ, n - 1)))   
            
        x = ITensors.dim(inds(S)[1])

        @assert(x == ITensors.dim(linkind(ϕ, n - 1)))
        
        r = uniqueinds(Vϕ, S) # Indices of density matrix
        lϕ = commonind(S, Vϕ) # Inner link index from density matrix diagonalization


        # Compute the theoretical maximum bond dimension that the enriched state cannot exceed:
        abs_maxdim = bipart_maxdim(s,n - 1) - ITensors.dim(lϕ)
        # Compute the number of eigenvectors of ɸ's projected density matrix to retain:
        Kry_linkdim_vec = [ITensors.dim(linkind(ψᵢ, n - 1)) for ψᵢ in ψ⃗]


        ω_maxdim = min(sum(Kry_linkdim_vec),abs_maxdim)

        if ω_maxdim !== 0


            # Construct identity matrix
            ID = 1
            rdim = 1
            for iv in r
                IDv = ITensor(iv', dag(iv));
                rdim *= ITensors.dim(iv)
                for i in 1:ITensors.dim(iv)
                IDv[iv' => i, iv => i] = 1.0
                end      
                ID = ID*IDv
            end   


            P = ID - prime(Vϕ, r)*dag(Vϕ) # Projector on to null-space of ρψ   

            C = combiner(r) # Combiner for indices
            # Check that P is non-zero   
            if abs(tr(matrix(C'*P*dag(C)))) > 1e-10    


                Dp, Vp, spec_P = eigen(
                        P, r', r,
                        ishermitian=true,
                        tags="P space",
                        cutoff=1e-1,
                        maxdim=rdim-ITensors.dim(lϕ),             ###potentially wrong
                     #   kwargs...,
                    )

                lp = commonind(Dp,Vp)

                ##constructing VpρₙVp
                VpρₙVp = Vp*ρₙ        
                VpρₙVp = VpρₙVp*dag(Vp')
                chkP = abs(tr(matrix(VpρₙVp))) ##chkP

            else
                chkP = 0    
            end
        else
            chkP = 0
        end

        if chkP >1e-15
            Dₙ, Vₙ, spec =eigen(VpρₙVp, lp', lp;
                ishermitian=true,
                tags=tags(linkind(ψ⃗[1], n - 1)),
                cutoff=Kr_cutoff,
                maxdim=ω_maxdim,            
               # kwargs...,
            )

            Vₙ = Vp*Vₙ

            lₙ₋₁ = commonind(Dₙ, Vₙ)

            # Construct the direct sum isometry 
            V, lnew = directsum(Vϕ => lϕ, Vₙ => lₙ₋₁; tags = tags(linkind(ϕ, n - 1)))
        else
                V = Vϕ
                lnew = lϕ

        end
        @assert ITensors.dim(linkind(ϕ, n - 1)) - ITensors.dim(lϕ) <=0
        # Update the enriched state
        phi[n] = V


        # Compute the new density matrix for the ancillary states
        C⃗ₙ₋₁ = [ψ⃗[i][n - 1] * C⃗ₙ[i] * dag(V) for i in 1:Nₘₚₛ]   
        C⃗ₙ₋₁′ = [prime(Cₙ₋₁, (s[n - 1], lnew)) for Cₙ₋₁ in C⃗ₙ₋₁]    
        ρ⃗ₙ₋₁ = C⃗ₙ₋₁′ .* dag.(C⃗ₙ₋₁)
        ρₙ₋₁ = sum(ρ⃗ₙ₋₁)

        # compute the density matrix for the real state    
        Cϕ = ϕ[n - 1] * Cϕprev * dag(V)
        Cϕd = prime(Cϕ, (s[n - 1], lnew))
        ρϕ = Cϕd * dag(Cϕ) 


        Cϕprev = Cϕ
        C⃗ₙ = C⃗ₙ₋₁
        ρₙ = ρₙ₋₁

        end


        phi[1] = Cϕprev

        if normalise
            phi[1] = phi[1]/norm(phi)
        end
        
        return phi
    end
    
    function Krylov_states(ψ,P,DP;kwargs...)
        (;k1,τ_Krylov) = P
        (;s,H_MPO) = DP
        ishermitian = get(kwargs, :ishermitian, true)


        ##Create the first k Krylov states
        Id = MPO(s,"Id")
        Kry_op = 1
        if ishermitian
            try
                Kry_op = Id-im*τ_Krylov*H_MPO
            catch 
                Kry_op = H_MPO
            end
        else
            Kry_op = H_MPO
        end

        list = []
        term = copy(ψ)

        for i =1:k1-1
            term = noprime(Kry_op*term)
            term = term/norm(term)
            push!(list,term)
        end

        return list
    end


end