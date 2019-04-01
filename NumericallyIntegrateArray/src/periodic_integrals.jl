function gauss_quadrature(y::Array{<:Number},GLweights::AbstractArray{<:Real,1})
	# Assume that y is evaluated at the zeros
	# I = Σ wi*y(zi) where zi are the roots of Pl of the appropriate order
	return squeeze(sum(GLweights.*y,1),1) :: Array{<:Number}
end

function clenshaw_curtis_quadrature(y::Array{<:Real})::Array{<:Real}
	N = size(y,1)::Int64-1 # number of divisions

	g = (FFTW.r2r(y, FFTW.REDFT00, 1)/N) :: Array{<:Real}

	inty = zeros(size(y)[2:end]...) :: Array{Float64}

	for ind in eachindex(inty)
		remaining_indices = ind2sub(size(inty),ind)
		inty[ind] = g[1,remaining_indices...]
		for k=1:div(N+1,2)-1
			inty[ind] += 2g[2k+1,remaining_indices...]/(1-(2k)^2)
		end
	end
	return inty
end

function clenshaw_curtis_quadrature(y::Array{<:Complex})::Array{<:Complex}
	Ny = size(y,1)::Int64-1

	# compute the fft of the even, complex sequence
	slice = [1:size(y,dim) for dim in 2:ndims(y)]
	g = (fft(vcat(y,y[Ny:-1:2,slice...]),1)/2Ny ) :: Array{<:Complex}

	inty = zeros(Complex128,size(y)[2:end]...) :: Array{Complex128}

	for ind in eachindex(inty)
		remaining_indices = ind2sub(size(inty),ind)
		inty[ind] = 2g[1,remaining_indices...]
		for k=1:div(Ny+1,2)-1
			inty[ind] += 2(g[2k+1,remaining_indices...]+
				g[end-2k+1,remaining_indices...])/(1-(2k)^2)
		end
	end
	return inty
end



#####################################################################################
#####################################################################################
# Integral on and in sphere
#####################################################################################

function sphericalIntegral3D(y::Array{<:Number,3},
	r::Union{Real,AbstractArray{<:Real,1}}=1,
	θ::AbstractArray{<:Real,1}=linspace(0,π,size(y,2)),
	ϕ::AbstractArray{<:Real,1}=linspace(0,2π,size(y,3));
	θint::String="clenshaw_quadrature",even::String="avg",
	GLweights::Union{Void,AbstractArray{<:Real,1}} = nothing,
	ϕ2π::Bool=true):: Array{<:Number,0}

	if typeof(r) <: AbstractArray
		z = simps(r.^2 .* y,x=r,even=even) :: Array{<:Number,2}
	else
		dr = r
		z = dr^2 .* simps((1:size(y,1)).^2 .* y,dx=dr,even=even) :: Array{<:Number,2}
	end

	trapzSpherical2D(z,θ,ϕ,θint=θint,GLweights=GLweights,ϕ2π=ϕ2π) :: Array{<:Number,0}
end

function sphericalIntegral2D(y::Array{<:Number,2},
	θ::AbstractArray{<:Real,1}=linspace(0,π,size(y,1)),
	ϕ::AbstractArray{<:Real,1}=linspace(0,2π,size(y,2));
	θint::String="clenshaw_quadrature",
	GLweights::Union{Void,AbstractArray{<:Real,1}} = nothing,
	ϕ2π::Bool=true):: Array{<:Number,0}

	trapzSpherical2D(y,θ,ϕ,θint=θint,GLweights=GLweights,ϕ2π=ϕ2π)
end

function sphericalIntegral1D(y::Vector{<:Number},
	ϕ::AbstractArray{<:Real,1}=linspace(0,2π,size(y,1));
	ϕ2π::Bool=true):: Array{<:Number,0}

	trapzCircle(y,ϕ,ϕ2π=ϕ2π)
end

function sphericalIntegral(y::Array{<:Number,N},args...;kwargs...) where N
	@assert(N<=3,"Integrals above 3D are not defined")
	f = eval(parse("sphericalIntegral"*string(N)*"D"))
	return f(y,args...,kwargs...) :: Array{<:Number,0}
end

function cylindricalIntegral1D(y::Vector{<:Number},
	ϕ::AbstractArray{<:Real,1}=linspace(0,2π,size(y,1));
	ϕ2π::Bool=true):: Array{<:Number,0}

	trapzCircle(y,ϕ,ϕ2π=ϕ2π)
end

function cylindricalIntegral2D(y::Array{<:Number,2},
	r::Union{Real,AbstractArray{<:Real,1}}=1,
	θ::AbstractArray{<:Real,1}=linspace(0,2π,size(y,2));
	even::String="avg",ϕ2π::Bool=true):: Array{<:Number,0}

	if typeof(r) <: AbstractArray
		z = simps(r.* y,x=r,even=even) :: Vector{<:Number}
	else
		dr = r
		z = dr*simps((1:size(y,1)).* y,dx=dr,even=even) :: Vector{<:Number}
	end

	trapzCircle(z,θ,ϕ2π=ϕ2π) :: Array{<:Number,0}
end

function cylindricalIntegral3D(y::Array{<:Number,3},
	r::Union{Real,AbstractArray{<:Real,1}}=1,
	θ::AbstractArray{<:Real,1}=linspace(0,2π,size(y,2)),
	z::Union{Real,AbstractArray{<:Real,1}}=1;
	even::String="avg",ϕ2π::Bool=true):: Array{<:Number,0}

	if typeof(r) <: AbstractArray
		yθz = simps(r.* y,x=r,even=even) :: Array{<:Number,2}
	else
		dr = r
		yθz = dr*simps((1:size(y,1)).* y,dx=dr,even=even) :: Array{<:Number,2}
	end

	T = promote_type(eltype(y), eltype(θ))
	yz = zeros(T,size(y,3))  :: Array{<:Number,1}

	for ind in eachindex(yz)
		yz[ind] = trapzCircle(yθz[1:end,ind],θ,ϕ2π=ϕ2π)[1]
	end

	if typeof(z) <: AbstractArray
		y_int = simps(z.* yz,x=z,even=even) :: Array{<:Number,0}
	else
		dz = z
		y_int = dz*simps((1:size(yz,1)).* yz,dx=dz,even=even) :: Array{<:Number,0}
	end

	return y_int :: Array{<:Number,0}
end

function cylindricalIntegral(y::Array{<:Number,N},args...;kwargs...) where N
	@assert(N<=3,"Integrals above 3D are not defined")
	f = eval(parse("cylindricalIntegral"*string(N)*"D"))
	return f(y,args...,kwargs...) :: Array{<:Number,0}
end