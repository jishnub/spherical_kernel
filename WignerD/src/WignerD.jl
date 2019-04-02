module WignerD

using OffsetArrays, WignerSymbols, LinearAlgebra,Libdl
using PointsOnASphere

export Ylmn,Ylmatrix,djmn,djmatrix,BiPoSH_s0,BiPoSH,BSH,Jy_eigen

function djmatrix(j,θ;kwargs...)
	m_range=get(kwargs,:m_range,-j:j)
	n_range=get(kwargs,:n_range,-j:j)
	dj = zeros(m_range,n_range)
	djmatrix!(dj,j,θ;m_range=m_range,n_range=n_range,kwargs...)
	return dj
end

function Jy_eigen(j)
	A = coeffi(j)
	λ,v = eigen(A)
	# We know that the eigenvalues of Jy are m ∈ -j:j, so we can round λ to integers and gain accuracy
	λ = round.(λ)
	#sort the array
	if issorted(λ)
		v = OffsetArray(collect(transpose(v)),-j:j,-j:j)
		λ = OffsetArray(λ,-j:j)
	else
		p = sortperm(λ)
		v = OffsetArray(collect(transpose(v[:,p])),-j:j,-j:j)
		λ = OffsetArray(λ[p],-j:j)
	end
	return λ,v
end

function djmatrix!(dj,j,θ::Real;kwargs...)

	λ = get(kwargs,:λ,nothing)
	v = get(kwargs,:v,nothing)
	m_range=get(kwargs,:m_range,-j:j)
	n_range=get(kwargs,:n_range,-j:j)

	if isnothing(λ) || isnothing(v)
		λ,v = Jy_eigen(j)
	end

	# check if symmetry conditions allow the index to be evaluated
	inds_covered = OffsetArray(falses(length(m_range),length(n_range)),
					m_range,n_range)

	@inbounds for (m,n) in Base.Iterators.product(m_range,n_range)

		inds_covered[m,n] && continue

		dj_m_n = zero(ComplexF64)
		dj_m_n_πmθ = zero(ComplexF64)
		dj_n_m = zero(ComplexF64)

		@inbounds for 𝑈 in axes(λ,1)
			dj_m_n += cis(-λ[𝑈]*θ) * v[𝑈,m] * conj(v[𝑈,n])
			if m != n
				dj_n_m += cis(-λ[𝑈]*(-θ)) * v[𝑈,m] * conj(v[𝑈,n])
			end
			
			dj_m_n_πmθ += cis(-λ[𝑈]*(π-θ)) * v[𝑈,m] * conj(v[𝑈,n])
			
		end

		dj[m,n] = real(dj_m_n)
		inds_covered[m,n] = true
		if !iszero(m) && -m in m_range
			dj[-m,n] = real(dj_m_n_πmθ)*(-1)^(j+n)
			inds_covered[-m,n] = true
		end

		if !iszero(n) && -n in n_range
			dj[m,-n] = real(dj_m_n_πmθ)*(-1)^(j+m)
			inds_covered[m,-n] = true
		end

		if !(iszero(m) && iszero(n)) && -m in n_range && -n in m_range
			dj[-n,-m] = real(dj_m_n)
			inds_covered[-n,-m] = true
		end

		if  !iszero(n) && m !=n && -n in n_range && -m in m_range
			dj[-m,-n] = (-1)^(n+m) * real(dj_m_n)
			inds_covered[-m,-n] = true
		end

		# transpose
		if m != n && m in n_range && n in m_range
			dj[n,m] = real(dj_n_m)
			inds_covered[n,m] = true
		end
		
	end
end

djmatrix(j,x::SphericalPoint;kwargs...) = djmatrix(j,x.θ;kwargs...)
djmatrix(j,m,n,θ::Real;kwargs...) = djmatrix(j,θ,m_range=m:m,n_range=n:n;kwargs...)
djmatrix(j,m,n,x::SphericalPoint;kwargs...) = djmatrix(j,x.θ,m_range=m:m,n_range=n:n;kwargs...)

djmatrix!(dj,j,x::SphericalPoint;kwargs...) = djmatrix(dj,j,x.θ;kwargs...)
djmatrix!(dj,j,m,n,θ::Real;kwargs...) = djmatrix(dj,j,θ,m_range=m:m,n_range=n:n;kwargs...)
djmatrix!(dj,j,m,n,x::SphericalPoint;kwargs...) = djmatrix(dj,j,x.θ,m_range=m:m,n_range=n:n;kwargs...)

function Ylmatrix(l,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...)

	n_range=get(kwargs,:n_range,-1:1)

	dj_θ = djmatrix(l,θ;kwargs...,n_range=n_range)
	Y = zeros(ComplexF64,axes(dj_θ)...)
	Ylmatrix!(Y,dj_θ,l,(θ,ϕ);n_range=n_range,kwargs...,compute_d_matrix=false)

	return Y
end

function Ylmatrix(dj_θ,l,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...)

	n_range=get(kwargs,:n_range,-1:1)
	m_range = axes(dj_θ,1)

	Y = zeros(ComplexF64,m_range,n_range)
	Ylmatrix!(Y,dj_θ,l,(θ,ϕ);compute_d_matrix=false,n_range=n_range,kwargs...)

	return Y
end

function Ylmatrix!(Y,dj_θ,l,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...)

	n_range=get(kwargs,:n_range,-1:1)
	m_range = get(kwargs,:m_range,-l:l)

	if get(kwargs,:compute_d_matrix,false):: Bool
		djmatrix!(dj_θ,l,θ;kwargs...,n_range=n_range)
	end

	for (m,n) in Base.Iterators.product(m_range,n_range)
		Y[m,n] = √((2l+1)/4π) * dj_θ[m,n] * cis(m*ϕ)
	end
end

Ylmatrix(l,m,n,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...) = Ylmatrix(l,(θ,ϕ);kwargs...,m_range=m:m,n_range=n:n)
Ylmatrix(l,m,n,x::SphericalPoint;kwargs...) = Ylmatrix(l,(x.θ,x.ϕ);kwargs...,m_range=m:m,n_range=n:n)
Ylmatrix(l,x::SphericalPoint;kwargs...) = Ylmatrix(l,(x.θ,x.ϕ);kwargs...)

Ylmatrix!(Y,dj_θ,l,m,n,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...) = Ylmatrix(Y,dj_θ,l,(θ,ϕ);kwargs...,m_range=m:m,n_range=n:n)
Ylmatrix!(Y,dj_θ,l,m,n,x::SphericalPoint;kwargs...) = Ylmatrix(Y,dj_θ,l,(x.θ,x.ϕ);kwargs...,m_range=m:m,n_range=n:n)
Ylmatrix!(Y,dj_θ,l,x::SphericalPoint;kwargs...) = Ylmatrix!(Y,dj_θ,l,(x.θ,x.ϕ);kwargs...)

Ylmatrix(dj_θ,l,m,n,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...) = Ylmatrix(dj_θ,l,(θ,ϕ);kwargs...,m_range=m:m,n_range=n:n)
Ylmatrix(dj_θ,l,m,n,x::SphericalPoint;kwargs...) = Ylmatrix(dj_θ,l,(x.θ,x.ϕ);kwargs...,m_range=m:m,n_range=n:n)
Ylmatrix(dj_θ,l,x::SphericalPoint;kwargs...) = Ylmatrix(dj_θ,l,(x.θ,x.ϕ);kwargs...)

X(j,n) = sqrt((j+n)*(j-n+1))

function coeffi(j)
	N = 2j+1
	A = zeros(ComplexF64,N,N)

	A[1,2]=-X(j,-j+1)/2im
    A[N,N-1]=X(j,-j+1)/2im

    @inbounds for i in 2:N-1
	    A[i,i+1]=-X(j,-j+i)/2im
	    A[i,i-1]=X(j,j-i+2)/2im
	end

	return Hermitian(A)
end

##################################################################################################

# Only t=0
function BiPoSH_s0(ℓ₁,ℓ₂,s::Integer,β::Integer,γ::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};
	Y_ℓ₁=zeros(0:-1,0:-1),Y_ℓ₂=zeros(0:-1,0:-1))
	# only t=0
	if iszero(length(Y_ℓ₁)) 
		Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁),n_range=β:β) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end
	if iszero(length(Y_ℓ₂))
		Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂),n_range=γ:γ) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end
	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied")
	m_max = min(ℓ₁,ℓ₂) ::Integer

	Y_BSH = zeros(ComplexF64,s:s,β:β,γ:γ)

	@inbounds for m in -m_max:m_max
		Y_BSH[s,β,γ] += clebschgordan(ℓ₁,m,ℓ₂,-m,s,0)*Y_ℓ₁[m,β]*Y_ℓ₂[-m,γ]
	end

	return Y_BSH
end

function BiPoSH_s0(ℓ₁,ℓ₂,s_range::AbstractRange,β::Integer,γ::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};wig3j_fn_ptr=nothing,
	Y_ℓ₁=zeros(0:-1,0:-1),Y_ℓ₂=zeros(0:-1,0:-1))
	# only t=0

	if iszero(length(Y_ℓ₁)) 
		Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁),n_range=β:β) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end
	if iszero(length(Y_ℓ₂))
		Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂),n_range=γ:γ) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end
	m_max = min(ℓ₁,ℓ₂)

	s_valid = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_intersection = intersect(s_range,s_valid)

	Y_BSH = zeros(ComplexF64,s_intersection,β:β,γ:γ)

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	@inbounds for m in -m_max:m_max
		CG = CG_ℓ₁mℓ₂nst(ℓ₁,m,ℓ₂;wig3j_fn_ptr=wig3j_fn_ptr)

		s_intersection = intersect(axes(Y_BSH,1),axes(CG,1))
		
		@inbounds for s in s_intersection
			Y_BSH[s,β,γ] += CG[s]*Y_ℓ₁[m,β]*Y_ℓ₂[-m,γ]
		end
	end

	if !isnothing(lib)
		Libdl.dlclose(lib)
	end

	return Y_BSH
end

function BiPoSH_s0(ℓ₁,ℓ₂,s::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};
	Y_ℓ₁=zeros(0:-1,0:-1),Y_ℓ₂=zeros(0:-1,0:-1))

	# only t=0
	if iszero(length(Y_ℓ₁))
		Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁)) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end

	if iszero(length(Y_ℓ₂)) 
		Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂)) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end

	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied")
	m_max = min(ℓ₁,ℓ₂)

	Y_BSH = zeros(ComplexF64,s:s,-1:1,-1:1)

	@inbounds for (s,β,γ) in Iterators.product(axes(Y_BSH)...),m in -m_max:m_max
		Y_BSH[s,β,γ] += clebschgordan(ℓ₁,m,ℓ₂,-m,s,0)*Y_ℓ₁[m,β]*Y_ℓ₂[-m,γ]
	end

	return Y_BSH
end

function BiPoSH_s0(ℓ₁,ℓ₂,s_range::AbstractRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};wig3j_fn_ptr=nothing,
	Y_ℓ₁=zeros(0:-1,0:-1),Y_ℓ₂=zeros(0:-1,0:-1))

	if iszero(length(Y_ℓ₁))
		Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁)) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end

	if iszero(length(Y_ℓ₂)) 
		Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂)) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end

	m_max = min(ℓ₁,ℓ₂)

	s_valid = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_intersection = intersect(s_valid,s_range)

	Y_BSH = zeros(ComplexF64,s_intersection,-1:1,-1:1)

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	@inbounds  for m in -m_max:m_max
		CG = CG_ℓ₁mℓ₂nst(ℓ₁,m,ℓ₂;wig3j_fn_ptr=wig3j_fn_ptr)

		s_intersection = intersect(axes(Y_BSH,1),axes(CG,1))

		@inbounds for (s,β,γ) in Iterators.product(s_intersection,axes(Y_BSH)[2:3]...)
			Y_BSH[s,β,γ] += CG[s]*Y_ℓ₁[m,β]*Y_ℓ₂[-m,γ]
		end
	end

	if !isnothing(lib)
		Libdl.dlclose(lib)
	end

	return Y_BSH
end

BiPoSH_s0(ℓ₁,ℓ₂,s,β::Integer,γ::Integer,
	x::SphericalPoint,x2::SphericalPoint;kwargs...) = BiPoSH_s0(ℓ₁,ℓ₂,s,β,γ,(x.θ,x.ϕ),(x2.θ,x2.ϕ);kwargs...)

BiPoSH_s0(ℓ₁,ℓ₂,s,
	x::SphericalPoint,x2::SphericalPoint;kwargs...) = BiPoSH_s0(ℓ₁,ℓ₂,s,(x.θ,x.ϕ),(x2.θ,x2.ϕ);kwargs...)

# Any t

struct BSH{N}
	smin :: Int64
	smax :: Int64
	arr :: OffsetArray{ComplexF64,N,Array{ComplexF64,N}}
end

function BSH(smin::Integer,smax::Integer,args...) 
	BSH(smin,smax,zeros(ComplexF64,1:((smax+1)^2-smin^2),args...))
end

BSH(s_range::UnitRange{<:Integer},args...) = BSH(first(s_range),last(s_range),
												zeros(ComplexF64,
												1:((last(s_range)+1)^2-first(s_range)^2),args...))

onedindex(s,t,smin=0) = s^2 - smin^2+(t+s)+1
onedindex(a::BSH,s,t) = onedindex(s,t,a.smin)

Base.getindex(a::BSH,s,t,args...) = a.arr[onedindex(a,s,t),args...]
Base.setindex!(a::BSH,x,s,t,args...) = a.arr[onedindex(a,s,t),args...] = x

Base.fill!(a::BSH,x) = fill!(a.arr,x)

function BiPoSH(ℓ₁,ℓ₂,s::Integer,t::Integer,β::Integer,γ::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real})

	Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁),n_range=β:β)
	Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂),n_range=γ:γ)
	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied")
	@assert(abs(t)<=s,"abs(t)<=s not satisfied")

	Y_BSH = BSH(s:s,β:β,γ:γ)

	@inbounds for m in -ℓ₁:ℓ₁
		n = t - m
		if abs(n) > ℓ₂
			continue
		end
		Y_BSH[s,t,β,γ] += clebschgordan(ℓ₁,m,ℓ₂,n,s,t)*Y_ℓ₁[m,β]*Y_ℓ₂[n,γ]
	end

	return OffsetArray(reshape([Y_BSH[s,t,β,γ]],1,1,1,1),s:s,t:t,β:β,γ:γ)
end

function BiPoSH(ℓ₁,ℓ₂,s_range::AbstractRange,β::Integer,γ::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};wig3j_fn_ptr=nothing)
	
	Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁),n_range=β:β)
	Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂),n_range=γ:γ)
	
	s_valid = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_intersection = intersect(s_range,s_valid)

	Y_BSH = BSH(s_intersection,β:β,γ:γ)
	t_max = Y_BSH.smax

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	for t=-t_max:t_max,m in -ℓ₁:ℓ₁
		n = t - m
		if abs(n) > ℓ₂
			continue
		end
		CG = CG_ℓ₁mℓ₂nst(ℓ₁,m,ℓ₂,t;wig3j_fn_ptr=wig3j_fn_ptr)

		s_intersection = intersect(axes(Y_BSH,1),axes(CG,1))
		
		for s in s_intersection
			Y_BSH[s,t,β,γ] += CG[s]*Y_ℓ₁[m,β]*Y_ℓ₂[n,γ]
		end
	end

	if !isnothing(lib)
		Libdl.dlclose(lib)
	end

	return Y_BSH
end

function BiPoSH(ℓ₁,ℓ₂,s::Integer,t::Integer,(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real})
	
	Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁))
	Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂))
	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied")
	@assert(abs(t)<=s,"abs(t)<=s not satisfied")

	Y_BSH = BSH(s:s,-1:1,-1:1)

	for β=-1:1,γ=-1:1
		for m in -ℓ₁:ℓ₁
			n = t - m
			if abs(n)>ℓ₂
				continue
			end
			Y_BSH[s,t,β,γ] += clebschgordan(ℓ₁,m,ℓ₂,n,s,t)*Y_ℓ₁[m,β]*Y_ℓ₂[n,γ]
		end
	end

	return OffsetArray(reshape([Y_BSH[s,t,β,γ]],1,1,1,1),s:s,t:t,β:β,γ:γ)
end

function BiPoSH(ℓ₁,ℓ₂,s_range::AbstractRange,(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};wig3j_fn_ptr=nothing)
	Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁))
	Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂))

	s_valid = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_valid = intersect(s_valid,s_range)

	Y_BSH = BSH(s_valid,-1:1,-1:1)
	t_max = Y_BSH.smax

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	for β in -1:1, γ in -1:1, t in -t_max:t_max, m in -ℓ₁:ℓ₁
		
		n = t - m
		if abs(n) > ℓ₂
			continue
		end
		CG = CG_ℓ₁mℓ₂nst(ℓ₁,m,ℓ₂,t;wig3j_fn_ptr=wig3j_fn_ptr)

		for s in intersect(s_valid,axes(CG,1))
			Y_BSH[s,t,β,γ] += CG[s]*Y_ℓ₁[m,β]*Y_ℓ₂[n,γ]
		end
	end

	if !isnothing(lib)
		Libdl.dlclose(lib)
	end

	return Y_BSH
end

BiPoSH(ℓ₁,ℓ₂,s,t,β::Integer,γ::Integer,x::SphericalPoint,x2::SphericalPoint) = BiPoSH(ℓ₁,ℓ₂,s,t,β,γ,(x.θ,x.ϕ),(x2.θ,x2.ϕ))
BiPoSH(ℓ₁,ℓ₂,s_range::AbstractRange,β::Integer,γ::Integer,x::SphericalPoint,x2::SphericalPoint) = BiPoSH(ℓ₁,ℓ₂,s_range,β,γ,(x.θ,x.ϕ),(x2.θ,x2.ϕ))
BiPoSH(ℓ₁,ℓ₂,s,t,x::SphericalPoint,x2::SphericalPoint) = BiPoSH(ℓ₁,ℓ₂,s,t,(x.θ,x.ϕ),(x2.θ,x2.ϕ))
BiPoSH(ℓ₁,ℓ₂,s_range::AbstractRange,x::SphericalPoint,x2::SphericalPoint) = BiPoSH(ℓ₁,ℓ₂,s_range,(x.θ,x.ϕ),(x2.θ,x2.ϕ))


##################################################################################################

function Wigner3j(j2,j3,m2,m3;wig3j_fn_ptr=nothing)
	
	m2,m3 = Int32(m2),Int32(m3)
	m1 = Int32(-(m2 + m3))

	j2,j3 = Int32(j2),Int32(j3)
	len = Int32(j2+j3+1)

	exitstatus = zero(Int32)

	w3j = zeros(Float64,len)

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	ccall(wig3j_fn_ptr,Cvoid,
		(Ref{Float64}, 	#w3j
			Ref{Int32},	#len
			# Ref{Int32},	#jmin
			# Ref{Int32},	#jmax
			Ref{Int32},	#j2
			Ref{Int32},	#j3
			Ref{Int32},	#m1
			Ref{Int32},	#m2
			Ref{Int32},	#m3
			Ref{Int32}),#exitstatus
		w3j,len, j2, j3, m1, m2,m3, exitstatus)

	if !isnothing(lib)
		Libdl.dlclose(lib)
	end

	return w3j
end

function Wigner3j!(w3j,j2,j3,m2,m3;wig3j_fn_ptr=nothing)
	
	m2,m3 = Int32(m2),Int32(m3)
	m1 = Int32(-(m2 + m3))

	j2,j3 = Int32(j2),Int32(j3)
	len = Int32(j2+j3+1)

	@assert(length(w3j)>=len,"length of output array must be atleast j2+j3+1=$(j2+j3+1),"*
							" supplied output array has a length of $(length(w3j))")

	exitstatus = zero(Int32)

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	ccall(wig3j_fn_ptr,Cvoid,
		(Ref{Float64}, 	#w3j
			Ref{Int32},	#len
			# Ref{Int32},	#jmin
			# Ref{Int32},	#jmax
			Ref{Int32},	#j2
			Ref{Int32},	#j3
			Ref{Int32},	#m1
			Ref{Int32},	#m2
			Ref{Int32},	#m3
			Ref{Int32}),#exitstatus
		w3j,len, j2, j3, m1, m2,m3, exitstatus)

	if !isnothing(lib)
		Libdl.dlclose(lib)
	end
end

function CG_ℓ₁mℓ₂nst(ℓ₁,m,ℓ₂,t=0;wig3j_fn_ptr=nothing)
	n = t-m
	smin = max(abs(ℓ₁-ℓ₂),abs(t))
	smax = ℓ₁ + ℓ₂
	w = Wigner3j(ℓ₁,ℓ₂,m,n;wig3j_fn_ptr=wig3j_fn_ptr)
	CG = OffsetArray(w[1:(smax-smin+1)],smin:smax)
	@inbounds for s in axes(CG,1)
		CG[s] *= √(2s+1)*(-1)^(ℓ₁-ℓ₂)
	end
	return CG
end

include("./precompile.jl")

end

