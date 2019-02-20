using WignerD,PointsOnASphere,TwoPointFunctions,Legendre
using OffsetArrays

function Yll_Pl_test(;ℓmax=10)
	n1 = Point2D(π/2,0)
	n2 = Point2D(π/2,π/3)
	Yℓℓ_00_10 = OffsetArray{ComplexF64,2}(undef,0:1,1:ℓmax)
	P = Pl_dPl(cosχ(n1,n2),ℓmax=ℓmax)
	
	for ℓ in axes(Yℓℓ_00_10,2)
		Yℓℓ_00_10[:,ℓ] = P[ℓ,:]
	end
	
	for ℓ in axes(Yℓℓ_00_10,2)
		Yℓℓ_00_10[0,ℓ] *= (-1)^ℓ/4π * √(2ℓ+1) 
		
		if ℓ==0
			continue
		end

		Yℓℓ_00_10[1,ℓ] *= im*(-1)^ℓ/4π * √(3*(2ℓ+1)/(ℓ*(ℓ+1)))  * ∂ϕ₂cosχ(n1,n2)
	end

	YB_00_10 = OffsetArray{ComplexF64}(undef,0:1,1:ℓmax)
	# for ℓ in 1:ℓmax 
	# 	YB_00_10[:,ℓ] .= BiPoSH_s0(ℓ,ℓ,0:1,0,0,n1,n2)[:,0,0]
	# end

	return Yℓℓ_00_10≈YB_00_10

end