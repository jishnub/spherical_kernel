module Sphere_2point_functions

using Reexport
include("./points_on_a_sphere.jl")
@reexport using PointsOnASphere

using ForwardDiff


export cosχ


cosχ(x::Vector{<:Real}) = cos(x[1])cos(x[3]) + sin(x[1])sin(x[3])cos(x[2]-x[4])

∂cosχ(x) = ForwardDiff.gradient(cosχ,x)
∂²cosχ(x) = ForwardDiff.hessian(cosχ,x)

for fn in ["cosχ","∂cosχ","∂²cosχ"]
	@eval $(Symbol(fn))((θ₁,ϕ₁)::NTuple{2,Real},(θ₂,ϕ₂)::NTuple{2,Real}) = $(Symbol(fn))([θ₁,ϕ₁,θ₂,ϕ₂])
	@eval $(Symbol(fn))(θ₁::Real,ϕ₁::Real,θ₂::Real,ϕ₂::Real) = $(Symbol(fn))((θ₁,ϕ₁),(θ₂,ϕ₂))
	@eval $(Symbol(fn))((θ₁,ϕ₁)::NTuple{2,Real},θ₂::Real,ϕ₂::Real) = $(Symbol(fn))((θ₁,ϕ₁),(θ₂,ϕ₂))
	@eval $(Symbol(fn))(θ₁::Real,ϕ₁::Real,(θ₂,ϕ₂)::NTuple{2,Real}) = $(Symbol(fn))((θ₁,ϕ₁),(θ₂,ϕ₂))
	@eval $(Symbol(fn))((θ₁,ϕ₁)::NTuple{2,Real},n::SphericalPoint) = $(Symbol(fn))((θ₁,ϕ₁),(n.θ,n.ϕ))
	@eval $(Symbol(fn))(θ₁::Real,ϕ₁::Real,n::SphericalPoint) = $(Symbol(fn))((θ₁,ϕ₁),(n.θ,n.ϕ))
	@eval $(Symbol(fn))(n::SphericalPoint,(θ₂,ϕ₂)::NTuple{2,Real}) = $(Symbol(fn))((n.θ,n.ϕ),(θ₂,ϕ₂))
	@eval $(Symbol(fn))(n::SphericalPoint,θ₂::Real,ϕ₂::Real) = $(Symbol(fn))((n.θ,n.ϕ),(θ₂,ϕ₂))
	@eval $(Symbol(fn))(n₁::SphericalPoint,n₂::SphericalPoint) = $(Symbol(fn))((n₁.θ,n₁.ϕ),(n₂.θ,n₂.ϕ))
end

# use subscripted variables
subscripts = Dict(1=>'\u2081',2=>'\u2082') # \_1 and \_2
powers = Dict(2=>'\u00B2',3=>'\u00B3')
for n in 4:9
	powers[n] = '\u2070' + n
end

# First derivatives
for pt in 1:2, (coord_ind,coord) in enumerate(["θ","ϕ"])
	fname = Symbol("∂$(coord)$(subscripts[pt])cosχ")
	flat_ind = (pt-1)*2 + coord_ind 
	@eval ($fname)(x) = ∂cosχ(x)[$flat_ind]
	@eval ($fname)(x...) = ∂cosχ(x...)[$flat_ind]
	@eval export $fname
end

# Second derivatives
for pt1 in 1:2, (coord1_ind,coord1) in enumerate(["θ","ϕ"])
	for pt2 in 1:2, (coord2_ind,coord2) in enumerate(["θ","ϕ"])
		fname = Symbol("∂$(coord1)$(subscripts[pt1])∂$(coord2)$(subscripts[pt2])cosχ")
		coord1_flat_ind = (pt1-1)*2 + coord1_ind
		coord2_flat_ind = (pt2-1)*2 + coord2_ind
		@eval ($fname)(x) = ∂²cosχ(x)[$coord1_flat_ind,$coord2_flat_ind]
		@eval ($fname)(x...) = ∂²cosχ(x...)[$coord1_flat_ind,$coord2_flat_ind]
		@eval export $fname
	end
end

# Second θ-derivative wrt one point
for pt=1:2
	@eval $(Symbol("∂²θ$(subscripts[pt])cosχ"))((θ₁,ϕ₁),(θ₂,ϕ₂)) = $(Symbol("∂θ$(subscripts[pt])∂θ$(subscripts[pt])cosχ"))((θ₁,ϕ₁),(θ₂,ϕ₂))
	@eval export $(Symbol("∂²θ$(subscripts[pt])cosχ"))
end

# The following functions are not defined if the gradients are evaluated at the poles
list_of_functions = []

for pt=1:2
	append!(list_of_functions,["∇ϕ$(subscripts[pt])cosχ","∇ϕ$(subscripts[pt])∂ϕ$(subscripts[pt])cosχ",
				"∂ϕ$(subscripts[pt])∇ϕ$(subscripts[pt])cosχ","∂²θ$(subscripts[pt])cosχ"])
end

for pt in 1:2
	@eval $(Symbol("∂²ϕ$(subscripts[pt])cosχ"))(x) = $(Symbol("∂ϕ$(subscripts[pt])∂ϕ$(subscripts[pt])cosχ"))(x)
	@eval $(Symbol("∂²ϕ$(subscripts[pt])cosχ"))(x...) = $(Symbol("∂ϕ$(subscripts[pt])∂ϕ$(subscripts[pt])cosχ"))(x...)
	
	@eval $(Symbol("∇ϕ$(subscripts[pt])cosχ"))((θ₁,ϕ₁)::NTuple{2,Real},(θ₂,ϕ₂)::NTuple{2,Real}) = $(Symbol("∂ϕ$(subscripts[pt])cosχ"))((θ₁,ϕ₁),(θ₂,ϕ₂))/sin($(Symbol("θ$(subscripts[pt])")))
	@eval $(Symbol("∇ϕ$(subscripts[pt])∂ϕ$(subscripts[pt])cosχ"))((θ₁,ϕ₁)::NTuple{2,Real},(θ₂,ϕ₂)::NTuple{2,Real}) = $(Symbol("∂²ϕ$(subscripts[pt])cosχ"))((θ₁,ϕ₁),(θ₂,ϕ₂))/sin($(Symbol("θ$(subscripts[pt])")))
	@eval $(Symbol("∂ϕ$(subscripts[pt])∇ϕ$(subscripts[pt])cosχ"))((θ₁,ϕ₁)::NTuple{2,Real},(θ₂,ϕ₂)::NTuple{2,Real}) = $(Symbol("∂²ϕ$(subscripts[pt])cosχ"))((θ₁,ϕ₁),(θ₂,ϕ₂))/sin($(Symbol("θ$(subscripts[pt])")))
	
	@eval export $(Symbol("∂ϕ$(subscripts[pt])∇ϕ$(subscripts[pt])cosχ")), $(Symbol("∇ϕ$(subscripts[pt])∂ϕ$(subscripts[pt])cosχ")), $(Symbol("∇ϕ$(subscripts[pt])cosχ")), $(Symbol("∂²ϕ$(subscripts[pt])cosχ"))
end

# Higher-order derivatives wrt. ϕ
for pt in 1:2
	for order in 3:2:9
		fn = "∂$(powers[order])ϕ$(subscripts[pt])cosχ"
		@eval $(Symbol(fn))(x) = $(Symbol("∂ϕ$(subscripts[pt])cosχ"))(x) * real(im^($order-1))
		@eval $(Symbol(fn))(x...) = $(Symbol("∂ϕ$(subscripts[pt])cosχ"))(x...) * real(im^($order-1))
		@eval export $(Symbol(fn))
		append!(list_of_functions,fn)
	end
	for order in 4:2:8
		fn = "∂$(powers[order])ϕ$(subscripts[pt])cosχ"
		@eval $(Symbol(fn))(x) = $(Symbol("∂²ϕ$(subscripts[pt])cosχ"))(x) * real(im^($order-2))
		@eval $(Symbol(fn))(x...) = $(Symbol("∂²ϕ$(subscripts[pt])cosχ"))(x...) * real(im^($order-2))
		@eval export $(Symbol(fn))
		append!(list_of_functions,fn)
	end
end

# Add methods
for pt in 1:2, fn in list_of_functions
	
	@eval $(Symbol(fn))(θ₁::Real,ϕ₁::Real,θ₂::Real,ϕ₂::Real) = $(Symbol(fn))((θ₁,ϕ₁),(θ₂,ϕ₂))
	@eval $(Symbol(fn))((θ₁,ϕ₁)::NTuple{2,Real},θ₂::Real,ϕ₂::Real) = $(Symbol(fn))((θ₁,ϕ₁),(θ₂,ϕ₂))
	@eval $(Symbol(fn))(θ₁::Real,ϕ₁::Real,(θ₂,ϕ₂)::NTuple{2,Real}) = $(Symbol(fn))((θ₁,ϕ₁),(θ₂,ϕ₂))
	@eval $(Symbol(fn))((θ₁,ϕ₁)::NTuple{2,Real},n::SphericalPoint) = $(Symbol(fn))((θ₁,ϕ₁),(n.θ,n.ϕ))
	@eval $(Symbol(fn))(θ₁::Real,ϕ₁::Real,n::SphericalPoint) = $(Symbol(fn))((θ₁,ϕ₁),(n.θ,n.ϕ))
	@eval $(Symbol(fn))(n::SphericalPoint,(θ₂,ϕ₂)::NTuple{2,Real}) = $(Symbol(fn))((n.θ,n.ϕ),(θ₂,ϕ₂))
	@eval $(Symbol(fn))(n::SphericalPoint,θ₂::Real,ϕ₂::Real) = $(Symbol(fn))((n.θ,n.ϕ),(θ₂,ϕ₂))
	@eval $(Symbol(fn))(n₁::SphericalPoint,n₂::SphericalPoint) = $(Symbol(fn))((n₁.θ,n₁.ϕ),(n₂.θ,n₂.ϕ))
end



end