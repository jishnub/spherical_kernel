module WignerD
using OffsetArrays, WignerSymbols, LinearAlgebra

# using PyCall
# @pyimport pyshtools.utils as SHTools

using PointsOnASphere

function djmatrix(j,θ;m_range=-j:j,n_range=-j:j)
	N = 2j+1
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

	dj = OffsetArray{Float64}(undef,m_range,n_range)

	# check if symmetry conditions allow the index to be evaluated
	inds_covered = OffsetArray(falses(size(dj)...),axes(dj)...)

	for (m,n) in Base.Iterators.product(axes(dj)...)

		if inds_covered[m,n]
			continue
		end

		dj_m_n = zero(ComplexF64)
		dj_m_n_πmθ = zero(ComplexF64)
		dj_n_m = zero(ComplexF64)

		for 𝑈 in axes(λ,1)
			dj_m_n += cis(-λ[𝑈]*θ) * v[𝑈,m] * conj(v[𝑈,n])
			if m != n
				dj_n_m += cis(-λ[𝑈]*(-θ)) * v[𝑈,m] * conj(v[𝑈,n])
			end
			
			dj_m_n_πmθ += cis(-λ[𝑈]*(π-θ)) * v[𝑈,m] * conj(v[𝑈,n])
			
		end

		# println("1 $m $n $(m) $(n)")
		dj[m,n] = real(dj_m_n)
		inds_covered[m,n] = true
		if ! iszero(m) && -m in m_range
			# println("2 $m $n $(-m) $(n)")
			dj[-m,n] = real(dj_m_n_πmθ)*(-1)^(j+n)
			inds_covered[-m,n] = true
		end

		if ! iszero(n) && -n in n_range
			# println("3 $m $n $(m) $(-n)")
			dj[m,-n] = real(dj_m_n_πmθ)*(-1)^(j+m)
			inds_covered[m,-n] = true
		end

		if !(iszero(m) && iszero(n)) && -m in n_range && -n in m_range
			# println("4 $m $n $(-n) $(-m)")
			dj[-n,-m] = real(dj_m_n)
			inds_covered[-n,-m] = true
		end

		if  !iszero(n) && m !=n && -n in n_range && -m in m_range
			# println("5 $m $n $(-m) $(-n)")
			dj[-m,-n] = (-1)^(n+m) * real(dj_m_n)
			inds_covered[-m,-n] = true
		end

		# transpose
		if m != n && m in n_range && n in m_range
			# println("6 $m $n $(n) $(m)")
			dj[n,m] = real(dj_n_m)
			inds_covered[n,m] = true
		end
		
	end

	return dj
end

djmatrix(j,m,n,θ) = djmatrix(j,θ,m_range=m:m,n_range=n:n)


Ylmatrix(l,m,n,(θ,ϕ)::Tuple{<:Real,<:Real}) = Ylmatrix(l,(θ,ϕ),m_range=m:m,n_range=n:n)
Ylmatrix(l,m,n,n1::Point2D) = Ylmatrix(l,(n1.θ,n1.ϕ),m_range=m:m,n_range=n:n)

function Ylmatrix(l,(θ,ϕ)::Tuple{<:Real,<:Real};m_range=-l:l,n_range=-1:1)

	dj_θ = djmatrix(l,θ,n_range=n_range,m_range=m_range)

	Y = OffsetArray{ComplexF64}(undef,axes(dj_θ)...)

	for (m,n) in Base.Iterators.product(axes(dj_θ)...)
		Y[m,n] = √((2l+1)/4π) * dj_θ[m,n] * cis(m*ϕ)
	end
	return Y
end

Ylmatrix(l,n::Point2D;m_range=-l:l,n_range=-1:1) = Ylmatrix(l,(n.θ,n.ϕ);m_range=m_range,n_range=n_range)

X(j,n) = sqrt((j+n)*(j-n+1))

function coeffi(j)
	N = 2j+1
	A = zeros(ComplexF64,N,N)
	# upper_diagonal = zeros(ComplexF64,N-1)

	A[1,2]=-X(j,-j+1)/2im
	# upper_diagonal[1] = -X(j,-j+1)/2im
    A[N,N-1]=X(j,-j+1)/2im

    for i in 2:N-1
    	# upper_diagonal[i] = -X(j,-j+i)/2im
	    A[i,i+1]=-X(j,-j+i)/2im
	    A[i,i-1]=X(j,j-i+2)/2im
	end

	# A = Hermitian(BandedMatrix(1=>upper_diagonal))

	return Hermitian(A)

	# return Hermitian(collect(A))
end


function BiPoSH_s0(ℓ₁,ℓ₂,s::Integer,β::Integer,γ::Integer,(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real})
	# only t=0
	Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁),n_range=β:β)
	Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂),n_range=γ:γ)
	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied")
	m_max = min(ℓ₁,ℓ₂)

	Y_BSH = OffsetArray(zeros(ComplexF64,1,1,1),s:s,β:β,γ:γ)

	for m in -m_max:m_max
		Y_BSH[s,β,γ] += clebschgordan(ℓ₁,m,ℓ₂,-m,s,0)*Y_ℓ₁[m,β]*Y_ℓ₂[-m,γ]
	end

	return Y_BSH
end

function BiPoSH_s0(ℓ₁,ℓ₂,s_range::AbstractRange,β::Integer,γ::Integer,(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real})
	# only t=0
	Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁),n_range=β:β)
	Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂),n_range=γ:γ)
	m_max = min(ℓ₁,ℓ₂)

	s_valid = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_intersection = intersect(s_range,s_valid)

	Y_BSH = OffsetArray(zeros(ComplexF64,length(s_intersection),1,1),s_intersection,β:β,γ:γ)

	for m in -m_max:m_max
		C_ℓ₁m_ℓ₂minusm_s0 = CG_tzero(ℓ₁,ℓ₂,m)

		s_intersection = intersect(axes(Y_BSH,1),axes(C_ℓ₁m_ℓ₂minusm_s0,1))
		
		for s in s_intersection
			Y_BSH[s,β,γ] += C_ℓ₁m_ℓ₂minusm_s0[s]*Y_ℓ₁[m,β]*Y_ℓ₂[-m,γ]
		end
	end

	return Y_BSH
end

function BiPoSH_s0(ℓ₁,ℓ₂,s::Integer,(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real})
	# only t=0
	Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁))
	Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂))
	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied")
	m_max = min(ℓ₁,ℓ₂)

	Y_BSH = OffsetArray(zeros(ComplexF64,1,3,3),s:s,-1:1,-1:1)

	for (s,β,γ) in Iterators.product(axes(Y_BSH)...),m in -m_max:m_max
		Y_BSH[s,β,γ] += clebschgordan(ℓ₁,m,ℓ₂,-m,s,0)*Y_ℓ₁[m,β]*Y_ℓ₂[-m,γ]
	end

	return Y_BSH
end

function BiPoSH_s0(ℓ₁,ℓ₂,s_range::AbstractRange,(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real})
	Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁))
	Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂))
	m_max = min(ℓ₁,ℓ₂)

	s_valid = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_intersection = intersect(s_valid,s_range)

	Y_BSH = OffsetArray(zeros(ComplexF64,length(s_intersection),3,3),s_intersection,-1:1,-1:1)

	for m in -m_max:m_max
		C_ℓ₁m_ℓ₂minusm_s0 = CG_tzero(ℓ₁,ℓ₂,m)

		s_intersection = intersect(axes(Y_BSH,1),axes(C_ℓ₁m_ℓ₂minusm_s0,1))

		for (s,β,γ) in Iterators.product(s_intersection,axes(Y_BSH)[2:3]...)
			Y_BSH[s,β,γ] += C_ℓ₁m_ℓ₂minusm_s0[s]*Y_ℓ₁[m,β]*Y_ℓ₂[-m,γ]
		end
	end

	return Y_BSH
end

BiPoSH_s0(ℓ₁,ℓ₂,s,β::Integer,γ::Integer,n1::Point2D,n2::Point2D) = BiPoSH_s0(ℓ₁,ℓ₂,s,β,γ,(n1.θ,n1.ϕ),(n2.θ,n2.ϕ))
BiPoSH_s0(ℓ₁,ℓ₂,s,n1::Point2D,n2::Point2D) = BiPoSH_s0(ℓ₁,ℓ₂,s,(n1.θ,n1.ϕ),(n2.θ,n2.ϕ))

function Wigner3j(j2,j3,m2,m3)
	
	m2,m3 = Int32(m2),Int32(m3)
	m1 = Int32(m2 + m3)

	j2,j3 = Int32(j2),Int32(j3)
	len = Int32(j2+j3+1)

	exitstatus = zero(Int32)

	w3j = zeros(Float64,len)

	ccall((:wigner3j_wrapper,"shtools_wrapper.so"),Cvoid,
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
	return w3j
end

function Wigner3j!(w3j,j2,j3,m2,m3)
	
	m2,m3 = Int32(m2),Int32(m3)
	m1 = Int32(m2 + m3)

	j2,j3 = Int32(j2),Int32(j3)
	len = Int32(j2+j3+1)

	@assert(length(w3j)>=len,"length of output array must be atleast j2+j3+1=$(j2+j3+1)")

	exitstatus = zero(Int32)

	ccall((:wigner3j_wrapper,"shtools_wrapper.so"),Cvoid,
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
end

function CG_tzero(ℓ₁,ℓ₂,m)
	smin = 0
	smax = ℓ₁ + ℓ₂
	w = Wigner3j(ℓ₁,ℓ₂,m,-m)
	CG = OffsetArray(w[1:(smax-smin+1)],smin:smax)
	for s in axes(CG,1)
		CG[s] *= √(2s+1)*(-1)^(ℓ₁-ℓ₂)
	end
	return CG
end

export Ylmn,Ylmatrix,djmn,djmatrix,BiPoSH_s0

end