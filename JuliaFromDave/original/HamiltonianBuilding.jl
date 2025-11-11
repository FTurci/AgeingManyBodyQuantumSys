module HamiltonianBuilding

export build_from_matrix,insert_sites,insert_ancilla

using ITensors

"""
    build_from_matrix(coefficents::AbstractMatrix, leftops, rightops,offset::Integer=0)::OpSum

Constructs an AutoMPO representation of the expresion Σ_{i,j} M_{i,j} L_i R_j where L_i and R_i are operators on site i and M_{i,j} is a matrix of coefficents

The operators can be spesifed either as stings or as an iterable of strings. If an iterable is given, then the result is the product of the operators in the iterable, all at the same site, in the given order.

The leftops, corresponding to the first index of the matrix, are always placed to the right of the rightops, corresponding to the second. 

When considering a subset of sites, which does not start at the first site, an offset can be given which shifts the index of the first site by the given amount.

#Examples
```
# Build coefficent matrix
m = [0.0 1.0
     1.0 0.0]


# Build hopping term
hopping_ops = build_from_matrix(m,"Cdag","C")

# Build interaction term
u = 0.1
interaction_ops = build_from_matrix(u * m, ["Cdag","C"], ["Cdag"."C"])

# Combine into a single Hamiltonian and build MPO
h = OpSum()
h += hopping_ops
h += interaction_ops

indices = siteinds("Fermion",2)
H = MPO(h,indices)
```
"""
function build_from_matrix(coefficents::AbstractMatrix, leftops, rightops;offset::Integer=0)::OpSum
    result = OpSum()

    for idx in eachindex(IndexCartesian(),coefficents)
        if coefficents[idx] == zero(eltype(coefficents))
            continue
        end

        i = idx[1] + offset
        j = idx[2] + offset

        term = [coefficents[idx]; [ [leftop, i] for leftop in leftops] |> Iterators.flatten |> collect ; [[rightop, j] for rightop in rightops] |> Iterators.flatten |> collect]
        result += tuple(term...)
    end

    return result
end

function build_from_matrix(coefficents::AbstractMatrix, leftop::AbstractString, rightop::AbstractString,args...;kwargs...)::OpSum
    build_from_matrix(coefficents, [leftop], [rightop],args...;kwargs...)
end

function build_from_matrix(coefficents::AbstractMatrix, leftop::AbstractString, rightop,args...;kwargs...)::OpSum
    build_from_matrix(coefficents, [leftop], rightop,args...;kwargs...)
end

function build_from_matrix(coefficents::AbstractMatrix, leftop,rightop::AbstractString,args...;kwargs...)::OpSum
    build_from_matrix(coefficents, leftop, [rightop],args...;kwargs...)
end


function insert_sites(op::Op, sites)::Op
    newsites = [site + count(i->i <= site, sites) for site in op.sites]
    Op(op.which_op,newsites...;op.params...)
end 

function insert_sites(ops::Prod,sites)::Prod
    Prod([insert_sites(op,sites) for op in ops])
end

function insert_sites(ops::Scaled,sites)::Scaled
    prefactor, prod = ops.args
    prefactor * insert_sites(prod,sites)
end

"""
    insert_sites(ops,sites)
Shifts the postions of the operators in autoMPO object ops to take account of additional sites being inserted at the positions given
in sites. The the positions given in sites refer to positions in the order before any insertions are made.

#Example
```
ops = OpSum()
ops += "C", 3, "Cdag",5
ops += 2.0, "N", 6

# Add extra sites at positions 1,4 and 6, so that site 3 must be moved
# 1 space to the right (to account for the addition at site 1), site
# 5 must be moved over 2 places (for the additions at sites 1 and 4)
# and position 6 must move to site 9 to account for the additions 
# at sites 1, 5 and the new 6.
insert_sites(ops,[1,4,6])
> sum(
    1.0 C(4,) Cdag(7)
    2.0 N(9,)
)
```

see also `insert_ancilla`
"""
function insert_sites(ops::OpSum,sites)::OpSum
    Sum([insert_sites(term,sites) for term in ops])
end

"""
    insert_ancilla(ops,system_sites)
Shifts positions of operators in autoMPO object ops to account for an ancilla site being inserted directly after 
each site in system_sites.

see also `insert_sites`
"""
function insert_ancilla(ops,system_sites)
    ancilla_positions = [i+1 for i in system_sites]
    insert_sites(ops,ancilla_positions)
end

end