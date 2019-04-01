###########################################################################################

# Regular grid for 1D array
function simps(y::AbstractVector,dx::Real=1;even="avg")

	if iseven(size(y,1))
		if !(even ∈ ("avg","first","last"))
			error("even has to be one of avg, first or last."*
				"Specified even is $(even)")
		end

		T = promote_type(eltype(y), Float64)
		result = zero(T)
		val = zero(T)
		first_ind = first(axes(y,1))
		last_ind = last(axes(y,1))

		if even ∈ ["avg","first"]
			result += simpsRegular1D(view(y,first_ind:last_ind-1),dx)
			val += trapzRegular1D(view(y,last_ind-1:last_ind),dx)
		end
		if even ∈ ["avg","last"]
			result += simpsRegular1D(view(y,first_ind+1:last_ind),dx)
			val += trapzRegular1D(view(y,first_ind:first_ind+1),dx)
		end
		if even == "avg"
			val /= 2
			result /=2
		end
		result += val
	else
		result = simpsRegular1D(y,dx)
	end
	return result 
end

# Regular grid for nD array
function simps(y::AbstractArray,dx::Real=1;even="avg")

	if iseven(size(y,1))
		if !(even ∈ ("avg","first","last"))
			error("even has to be one of avg, first or last."*
				"Specified even is $(even)")
		end

		T = promote_type(eltype(y), Float64)

		remaining_axes = CartesianIndices(axes(y)[2:end])
		result = zeros(T,remaining_axes.indices)
		val = zeros(T,remaining_axes.indices)

		first_ind = first(axes(y,1))
		last_ind = last(axes(y,1))

		if even ∈ ["avg","first"]
			result .+= simpsRegular1D(view(y,first_ind:last_ind-1,remaining_axes),dx)
			val .+= trapzRegular1D(view(y,last_ind-1:last_ind,remaining_axes),dx)
		end
		if even ∈ ["avg","last"]
			result .+= simpsRegular1D(view(y,first_ind+1:last_ind,remaining_axes),dx)
			val .+= trapzRegular1D(view(y,first_ind:first_ind+1,remaining_axes),dx)
		end
		if even == "avg"
			val ./= 2
			result ./=2
		end
		result .+= val
	else
		result = simpsRegular1D(y,dx)
	end
	return result 
end

# Irregular grid for 1D array
function simps(y::AbstractVector,x::AbstractVector{<:Real};even="avg")

	if iseven(size(y,1))
		if !(even ∈ ("avg","first","last"))
			error("even has to be one of avg, first or last."*
				"Specified even is $(even)")
		end

		T = promote_type(eltype(y), Float64)
		result = zero(T)
		val = zero(T)
		first_ind = first(axes(y,1))
		last_ind = last(axes(y,1))
		first_ind_x = first(axes(x,1))
		last_ind_x = last(axes(x,1))

		if even ∈ ["avg","first"]
			result += simpsIrregular1D(view(y,first_ind:last_ind-1),
				view(x,first_ind_x:last_ind_x-1))
			val += trapzRegular1D(view(y,last_ind-1:last_ind),
				x[last_ind_x]-x[last_ind_x-1])
		end
		if even ∈ ["avg","last"]
			result += simpsIrregular1D(view(y,first_ind+1:last_ind),
				view(x,first_ind_x+1:last_ind_x))
			val += trapzRegular1D(view(y,first_ind:first_ind+1),
				x[first_ind_x+1]-x[first_ind_x])
		end
		if even == "avg"
			val /= 2.
			result /=2.
		end
		result += val
	else
		result = simpsIrregular1D(y,x)
	end
	return result
end

# Irregular grid for nD array
function simps(y::AbstractArray,x::AbstractVector{<:Real};even="avg")

	if iseven(size(y,1))
		if !(even ∈ ("avg","first","last"))
			error("even has to be one of avg, first or last."*
				"Specified even is $(even)")
		end

		T = promote_type(eltype(y), Float64)
		remaining_axes = CartesianIndices(axes(y)[2:end])
		result = zeros(T,remaining_axes.indices)
		val = zeros(T,remaining_axes.indices)
		first_ind = first(axes(y,1))
		last_ind = last(axes(y,1))
		first_ind_x = first(axes(x,1))
		last_ind_x = last(axes(x,1))

		if even ∈ ["avg","first"]
			result .+= simpsIrregular1D(view(y,first_ind:last_ind-1,remaining_axes),
				view(x,first_ind_x:last_ind_x-1))
			val .+= trapzRegular1D(view(y,last_ind-1:last_ind,remaining_axes),
				x[last_ind_x]-x[last_ind_x-1])
		end
		if even ∈ ["avg","last"]
			result .+= simpsIrregular1D(view(y,first_ind+1:last_ind,remaining_axes),
				view(x,first_ind_x+1:last_ind_x))
			val .+= trapzRegular1D(view(y,first_ind:first_ind+1,remaining_axes),
				x[first_ind_x+1]-x[first_ind_x])
		end
		if even == "avg"
			val ./= 2.
			result ./=2.
		end
		result .+= val
	else
		result = simpsIrregular1D(y,x)
	end
	return result
end

#############################################################################

function simpsweight(Nelems,index)
	@assert(1<=index<=Nelems,"index not in range")
	if index==1 || index==Nelems 
		return 1
	elseif iseven(index)
		return 4
	end
	return 2
end

function quadratic_fit_integrate!(T,coeff,y,x)
	# Find the lagrange polynomial fit to y(x) (quadratic)
	# integrate this between first(x) and last(x)
	@assert(length(y)==length(x)==3)
	first_ind_x = first(axes(x,1))

	coeff[1] = -(x[first_ind_x]-x[first_ind_x+2])*
			(2x[first_ind_x]-3x[first_ind_x+1]+x[first_ind_x+2])/
			(6*(x[first_ind_x]-x[first_ind_x+1]))

	coeff[2] = -(x[first_ind_x]-x[first_ind_x+2])^3/
				(6*(x[first_ind_x]-x[first_ind_x+1])*
					(x[first_ind_x+1]-x[first_ind_x+2]))

	coeff[3] = (x[first_ind_x]-x[first_ind_x+2])*
				(x[first_ind_x]-3x[first_ind_x+1]+2x[first_ind_x+2])/
					(6*(x[first_ind_x+1]-x[first_ind_x+2]))

	s = zero(T)
	for (c_i,y_i) in zip(coeff,y)
		s += c_i*y_i
	end
	return s
end

#############################################################################
## Simpson's 1/3 rule
#############################################################################

# 1D array with uniform grid spacing for an odd number of points
function simpsRegular1D(y::AbstractVector,dx::Real=1)
	
	T = promote_type(eltype(y),Float64)
	int_y = zero(T)

	N = size(y,1)

	for (ind,y_i) in enumerate(y)
		w = simpsweight(N,ind)
		int_y += w*y_i
	end
	
	return int_y*dx/3
end

# UnitRanges can be integrated exactly
function simpsRegular1D(y::UnitRange{R},dx::Real=1) where R<:Real
	T = promote_type(R,Float64)
	(convert(T,y.stop)^2 - convert(T,y.start)^2)/2 * dx
end

# N dimensional array, integrate along first axis
function simpsRegular1D(y::AbstractArray,dx::Real=1)

	remaining_axes = CartesianIndices(axes(y)[2:end])
	T = promote_type(eltype(y),Float64)
	int_y = zeros(T,remaining_axes.indices)
	for ind in remaining_axes
		int_y[ind] = simpsRegular1D(view(y,:,ind),dx) 
	end
	return int_y

end

#############################################################################
## Simpson's 1/3 rule for irregular grids
#############################################################################

# 1D array with non-uniform grid spacing, integral along one axis
function simpsIrregular1D(y::AbstractVector,x::AbstractVector)

	@assert(isodd(length(y)),"The array should have an odd number of elements")
	@assert(length(x)==length(y),"y and x need to have the same number of elements")

	T = promote_type(eltype(y), Float64)
	int_y = zero(T)

	y_no_offset = OffsetArrays.no_offset_view(y)
	x_no_offset = OffsetArrays.no_offset_view(x)

	# The strategy is to fit a quadratic to three points, and compute its integral
	coeffs = zeros(T,3)
	for ind in 1:2:length(x)-2

		int_y += quadratic_fit_integrate!(T,coeffs,view(y_no_offset,ind:ind+2),
						view(x_no_offset,ind:ind+2))
	end

	return int_y
end

# N dimensional array, integrate along first axis
function simpsIrregular1D(y::AbstractArray,x::AbstractVector)

	remaining_axes = CartesianIndices(axes(y)[2:end])
	T = promote_type(eltype(y),Float64)
	int_y = zeros(T,remaining_axes.indices)
	for ind in remaining_axes
		int_y[ind] = simpsIrregular1D(view(y,:,ind),x) 
	end
	return int_y
end

function simpsCircle(y::AbstractVector,
	ϕ::AbstractArray{<:Real,1}=LinRange(0,2π,size(y,1));
	even::String="avg",ϕ2π::Bool=true)::Array{<:Number,0}

	if ϕ2π
		z = simps(y,x=ϕ,even=even)
	else

		info("Assuming that 2π is not included,"*
			" will append it to the array before integrating.\n"*
			"If 2π is included set ϕ2π=true in the function call")

		check_last_value(ϕ,2π) # check if the last value of ϕ is 2π

		push!(y,y[1])

		z = simps(y,x=push!(ϕ,2π),even=even)

	end

	return z 
end

function simpsSpherical2D(y::AbstractArray,
	θ::AbstractArray{<:Real,1}=LinRange(0,π,size(y,1)),
	ϕ::AbstractArray{<:Real,1}=LinRange(0,2π,size(y,2));
	even="avg",θint="clenshaw_quadrature",GLweights=nothing,
	ϕ2π=true)::Array{<:Number,0}

	if θint=="gauss_quadrature"
		
		if GLweights != nothing
			z = gauss_quadrature(y,GLweights)
		end

	elseif θint=="clenshaw_quadrature"
		
		z = clenshaw_curtis_quadrature(y)
		
	else
		z = simps(sin.(θ).*y,x=θ,even=even)
	end

	simpsCircle(z,ϕ,even=even,ϕ2π=ϕ2π) 
end

function simpsSpherical3D(y::AbstractArray,
	r::Union{Real,AbstractArray{<:Real,1}}=1,
	θ::AbstractArray{<:Real,1}=LinRange(0,π,size(y,2)),
	ϕ::AbstractArray{<:Real,1}=LinRange(0,π,size(y,3));
	even="avg",θint="clenshaw_quadrature",GLweights = nothing,
	ϕ2π=true)
	
	if typeof(r) <: AbstractArray
		z = simps(r.^2 .* y,x=r,even=even)
	else
		dr = r
		z = dr^2*simps((1:size(y,1)).^2 .* y,dx=dr,even=even)
	end

	simpsSpherical2D(z,θ,ϕ,θint=θint,
		GLweights=GLweights,even=even,ϕ2π=ϕ2π)
end