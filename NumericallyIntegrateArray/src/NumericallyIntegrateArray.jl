module NumericallyIntegrateArray
using OffsetArrays,Polynomials

# package code goes here
export simps,trapz
export sphericalIntegral,cylindricalIntegral

############################################################################################
## helper functions
############################################################################################

function check_last_value(a::AbstractArray,val=2π;var="ϕ",switch="ϕ2π")
	if a[end]≈val
		warn("$(var) appears to have $(val) as the last element, "*
			"but the keyword argument $(switch) is set to false.\n"*
			"To fix this, add $(switch)=true to your function call")
	end
end

scalar_to_0dim_array(z::Number)::Array{<:Number,0} = setindex!(Array{eltype(z)}(),z)
scalar_to_0dim_array(T::Type{<:Number},z::Number)::Array{<:Number,0} = setindex!(Array{T}(),z)

##############################################################################################

include("./trapz.jl")
include("./simps.jl")

#####################################################################################

end # module
