###########################################################################################
# Operators
###########################################################################################

module finite_difference

	using Reexport
	@reexport using SparseArrays

	function derivStencil(order::Integer,ptsleft::Integer,ptsright::Integer;gridspacing::Real=1)
		
		M = zeros(ptsleft + ptsright+1,ptsleft + ptsright+1)

		for (p_ind,p) in enumerate(0:(ptsleft + ptsright)) , (m_ind,m) in enumerate(-ptsleft:ptsright)
			M[p_ind,m_ind] = m^p/factorial(p)
		end

		M_order = inv(M)[:,order+1]./gridspacing^order
		
		if (ptsleft == ptsright) && isodd(order)
			# Fix loss of numerical precision
			M_order[ptsleft+1] = 0.
		end

		stencil = sparsevec(M_order)
	end

	function derivStencil(order::Integer,pts::Integer;kwargs...) 
		if isodd(pts)
			return derivStencil(order,div(pts,2),div(pts,2);kwargs...)
		else
			return derivStencil(order,div(pts,2)-1,div(pts,2);kwargs...)
		end
	end

	nextodd(n::Integer) = isodd(n) ? n+2 : n+1

	function ceilsearch(arr,n) 
		# assume sorted
		for elem in arr
			if elem>=n
				return elem
			end
		end
		return n
	end

	# Dictionary of gridpt => order of derivative upto that gridpt

	function D(N,stencil_gridpts=Dict(6=>2,42=>4);
		left_edge_npts=2,left_edge_ghost=false,right_edge_npts=2,right_edge_ghost=false,kwargs...)

		@assert(N≥2,"Need at least 2 points to compute the derivative")
		
		N_cols = N
		if left_edge_ghost
			N_cols += 1
		end
		if right_edge_ghost
			N_cols += 1
		end

		S = spzeros(N,N_cols)

		gridpt_cutoffs_list = sort(collect(keys(stencil_gridpts)))
		gridpt_cutoff_maxorder = maximum(keys(stencil_gridpts))
		maxorder = stencil_gridpts[gridpt_cutoff_maxorder]

		# Derivatives on the boundary

		if left_edge_ghost
			startpt = 2 - div(left_edge_npts,2)
			endpt = startpt+left_edge_npts-1
			if startpt >=1 
				S[1,startpt:endpt] = derivStencil(1,left_edge_npts;kwargs...)
			else
				S[1,1:left_edge_npts] = derivStencil(1,1,left_edge_npts-2;kwargs...)
			end
		else
			S[1,1:left_edge_npts] = derivStencil(1,0,left_edge_npts-1;kwargs...)
		end

		if right_edge_ghost
			endpt = N +  div(right_edge_npts,2) + ( left_edge_ghost ? 1 : 0 )
			startpt = endpt - right_edge_npts + 1
			if endpt<=N_cols
				S[end,startpt:endpt] = derivStencil(1,right_edge_npts;kwargs...)
			else
				S[end,(N_cols-right_edge_npts+1):N_cols] = derivStencil(1,right_edge_npts-2,1;kwargs...)
			end
		else
			S[end,(N_cols-right_edge_npts+1):N_cols] = derivStencil(1,right_edge_npts-1,0;kwargs...)
		end

		for gridpt in 2:N-1

			gridpt_cutoff =  ceilsearch(gridpt_cutoffs_list,min(gridpt,N-gridpt+1))
			if haskey(stencil_gridpts,gridpt_cutoff)
				npts = stencil_gridpts[gridpt_cutoff]
			else
				npts = nextodd(maxorder)
			end

			diagpt = gridpt + ( left_edge_ghost ? 1 : 0 )
			startpt = max(1,diagpt - div(npts,2)-(diagpt + div(npts,2) > N_cols ? diagpt + div(npts,2) - N_cols : 0 ))
			endpt = min(N_cols,startpt + npts -1)
			npts_left = diagpt - startpt
			npts_right = endpt - diagpt

			S[gridpt,startpt:endpt] = derivStencil(1,npts_left,npts_right;kwargs...)
		end

		return S
	end

	function D²(N,stencil_gridpts=Dict(6=>3,42=>5);
		left_edge_npts=3,left_edge_ghost=false,right_edge_npts=3,right_edge_ghost=false,kwargs...)

		@assert(N≥3,"Need at least 3 points to compute the second derivative")
		
		N_cols = N
		if left_edge_ghost
			N_cols += 1
		end
		if right_edge_ghost
			N_cols += 1
		end

		S = spzeros(N,N_cols)

		gridpt_cutoffs_list = sort(collect(keys(stencil_gridpts)))
		gridpt_cutoff_maxorder = maximum(keys(stencil_gridpts))
		maxorder = stencil_gridpts[gridpt_cutoff_maxorder]

		# Derivatives on the boundary

		if left_edge_ghost
			startpt = 2 - div(left_edge_npts,2)
			endpt = startpt+left_edge_npts-1
			if startpt >=1 
				S[1,startpt:endpt] = derivStencil(2,left_edge_npts;kwargs...)
			else
				S[1,1:left_edge_npts] = derivStencil(2,1,left_edge_npts-2;kwargs...)
			end
		else
			S[1,1:left_edge_npts] = derivStencil(2,0,left_edge_npts-1;kwargs...)
		end

		if right_edge_ghost
			endpt = N +  div(right_edge_npts,2) + ( left_edge_ghost ? 1 : 0 )
			startpt = endpt - right_edge_npts + 1
			if endpt<=N_cols
				S[end,startpt:endpt] = derivStencil(2,right_edge_npts;kwargs...)
			else
				S[end,(N_cols-right_edge_npts+1):N_cols] = derivStencil(2,right_edge_npts-2,1;kwargs...)
			end
		else
			S[end,(N_cols-right_edge_npts+1):N_cols] = derivStencil(2,right_edge_npts-1,0;kwargs...)
		end

		for gridpt in 2:N-1

			gridpt_cutoff =  ceilsearch(gridpt_cutoffs_list,min(gridpt,N-gridpt+1))
			if haskey(stencil_gridpts,gridpt_cutoff)
				npts = stencil_gridpts[gridpt_cutoff]
			else
				npts = nextodd(maxorder)
			end

			diagpt = gridpt + ( left_edge_ghost ? 1 : 0 )
			startpt = max(1,diagpt - div(npts,2)-(diagpt + div(npts,2) > N_cols ? diagpt + div(npts,2) - N_cols : 0 ))
			endpt = min(N_cols,startpt + npts -1)
			npts_left = diagpt - startpt
			npts_right = endpt - diagpt

			S[gridpt,startpt:endpt] = derivStencil(2,npts_left,npts_right;kwargs...)
		end

		return S
	end

	function Dⁿ(order,N,stencil_gridpts=Dict(6=>order+1,42=>order+3);
		left_edge_npts=order+1,left_edge_ghost=false,right_edge_npts=order+1,right_edge_ghost=false,kwargs...)

		@assert(N≥order+1,"Need at least $(order+1) points to compute the derivative")
		
		N_cols = N
		if left_edge_ghost
			N_cols += 1
		end
		if right_edge_ghost
			N_cols += 1
		end

		S = spzeros(N,N_cols)

		gridpt_cutoffs_list = sort(collect(keys(stencil_gridpts)))
		gridpt_cutoff_maxorder = maximum(keys(stencil_gridpts))
		maxorder = stencil_gridpts[gridpt_cutoff_maxorder]

		# Derivatives on the boundary

		if left_edge_ghost
			startpt = 2 - div(left_edge_npts,2)
			endpt = startpt+left_edge_npts-1
			if startpt >=1 
				S[1,startpt:endpt] = derivStencil(order,left_edge_npts;kwargs...)
			else
				S[1,1:left_edge_npts] = derivStencil(order,1,left_edge_npts-2;kwargs...)
			end
		else
			S[1,1:left_edge_npts] = derivStencil(order,0,left_edge_npts-1;kwargs...)
		end

		if right_edge_ghost
			endpt = N +  div(right_edge_npts,2) + ( left_edge_ghost ? 1 : 0 )
			startpt = endpt - right_edge_npts + 1
			if endpt<=N_cols
				S[end,startpt:endpt] = derivStencil(order,right_edge_npts;kwargs...)
			else
				S[end,(N_cols-right_edge_npts+1):N_cols] = derivStencil(order,right_edge_npts-2,1;kwargs...)
			end
		else
			S[end,(N_cols-right_edge_npts+1):N_cols] = derivStencil(order,right_edge_npts-1,0;kwargs...)
		end

		for gridpt in 2:N-1

			gridpt_cutoff =  ceilsearch(gridpt_cutoffs_list,min(gridpt,N-gridpt+1))
			if haskey(stencil_gridpts,gridpt_cutoff)
				npts = stencil_gridpts[gridpt_cutoff]
			else
				npts = nextodd(maxorder)
			end

			diagpt = gridpt + ( left_edge_ghost ? 1 : 0 )
			startpt = max(1,diagpt - div(npts,2)-(diagpt + div(npts,2) > N_cols ? diagpt + div(npts,2) - N_cols : 0 ))
			endpt = min(N_cols,startpt + npts -1)
			npts_left = diagpt - startpt
			npts_right = endpt - diagpt

			S[gridpt,startpt:endpt] = derivStencil(order,npts_left,npts_right;kwargs...)
		end

		return S
	end

	function dbydr(dr::AbstractArray,stencil_gridpts=Dict(6=>3,42=>5);kwargs...)
		return D(length(dr),stencil_gridpts;kwargs...) ./ dr
	end

	function dbydr(N::Integer,dx::Real,stencil_gridpts=Dict(6=>3,42=>5);kwargs...)
		return D(N,stencil_gridpts;kwargs...) ./ dx
	end

	function roll(arr,shift)
		# translation with wrap
		# A′ = T(shift) A
		# translation: A′(i) = A(i-shift)
		# translation with wrap: A′(i+1) = A(mod(i-shift,N)+1), i=0...N-1

		N = size(arr,ndims(arr))
		shift = mod(shift,N)
		if shift==0
			return arr
		end

		newarr = similar(arr)
		
		for i in 0:N-1
			newarr[:,:,i+1] .= arr[:,:,mod(i-shift,N)+1]
		end
		return newarr
	end

	export D,D²,Dⁿ,dbydr,derivStencil,roll
end

#################################################################
# Green function radial components, main function
#################################################################

module Greenfn_radial

	using Reexport
	using LinearAlgebra,SparseArrays,DelimitedFiles
	@reexport using Main.finite_difference
	@reexport using OffsetArrays, FITSIO,JLD2,
	FileIO,ParallelUtilities,Polynomials,
	Distributed,Printf,ProgressMeter

	export Rsun,nr,r,dr,ddr,c,ρ,g,N2,γ_damping,r_src_default
	export components_radial_source,all_components,Gfn_path_from_source_radius,Ω

	function load_solar_model()
		
		modelS_meta = readdlm("ModelS.meta",comments=true, comment_char='#');
		Msun,Rsun = modelS_meta[1:2];

		modelS = readdlm("ModelS",comments=true, comment_char='#');
		modelS_detailed = readdlm("ModelS.detailed",comments=true, comment_char='#');

		# HMIfreq=readdlm("Schou_freq181qr.1288");
		# ℓ_HMI,n_HMI,ν_HMI,γ_HMI=[HMIfreq[:,i] for i=1:4];

		# Fit modes above ℓ=11
		# γ_damping = polyfit(sort(ν_HMI[ℓ_HMI.>11]).*(2π*1e-6),sort(γ_HMI[ℓ_HMI.>11]).*(2π*1e-6),6)
		γ_damping = 2π*4e-6 # constant damping to compare with krishnendu
		
		r = modelS[:,1]*Rsun; flip = (length(r)-1):-1:1; # leave out the center to avoid singularities
		r = r[flip];
		nr = length(r);

		dr = D(nr)*r;
		ddr = dbydr(dr);
		c = modelS[flip,2];
		ρ = modelS[flip,3];

		G = 6.67428e-8 # cgs units
		m = Msun.*exp.(modelS_detailed[flip,2])

		g = @. G*m/r^2

		N2 = @. g * modelS_detailed[flip,15] / r

		return Rsun,nr,r,dr,ddr,c,ρ,g,N2,γ_damping
	end

	const Rsun,nr,r,dr,ddr,c,ρ,g,N2,γ_damping = load_solar_model()
	const r_src_default = Rsun - 75e5

	function source(ω,ℓ;r_src=r_src_default)

		r_src_ind = argmin(abs.(r.-r_src))
		r_src_on_grid = r[r_src_ind];

		σsrc = max(r_src_on_grid - r[max(1,r_src_ind-2)],r[min(nr,r_src_ind+2)] - r_src_on_grid)

		# println("rsrc $r_src_on_grid σ $σsrc")

		delta_fn = @. exp(-(r - r_src_on_grid)^2 / 2σsrc^2) / √(2π*σsrc^2) /r^2;

		Sr = append!(delta_fn[2:nr],zeros(nr-1))
		Sh_bottom = @. -√(ℓ*(ℓ+1))/ω^2 * delta_fn /(r*ρ)

		Sh = append!(zeros(nr-1),Sh_bottom[1:nr-1])

		return Sr,Sh;
	end

	function ℒr()

		L14 = dbydr(dr[2:nr-1],left_edge_ghost=true,left_edge_npts=3,right_edge_ghost=false,right_edge_npts=3) # (nr-2 x nr-1)
		L14[diagind(L14,1)] .+= (@. g/c^2)[2:nr-1]

		L22 = derivStencil(1,1,0,gridspacing=dr[end]) .+ sparsevec([2],[2/r[end] - g[end]/c[end]^2],2) # (2 x 1)

		L33 = derivStencil(1,0,1,gridspacing=dr[1]) .+ sparsevec([1],[g[1]/c[1]^2],2) # (2 x 1)

		L41 = dbydr(dr[2:end-1],left_edge_ghost=false,left_edge_npts=3,right_edge_ghost=true,right_edge_npts=3) # (nr-2 x nr-1)
		L41[diagind(L41,0)] .+= (@. 2/r - g/c^2)[2:end-1]

		M = spzeros(ComplexF64,2*(nr-1),2*(nr-1))

		M[diagind(M)[1:nr-2]] = @. (ρ*N2)[2:nr-1]

		M[1:nr-2,end-(nr-1)+1:end] = L14

		M[nr-1,nr-2:nr-1] = L22

		M[nr,nr:nr+1] = L33

		M[nr+1:end,1:nr-1] = L41

		M[diagind(M)[nr+1:end]] = @. (1/(ρ*c^2)[2:nr-1]) # Assign to diagonals

		return M
	end

	const M = ℒr()

	Ω(ℓ,N) = √((ℓ+N)*(ℓ-N+1)/2)

	function ℒωℓr(ω,ℓ)

		ω -= im*γ_damping
		# ω += im*γ_damping(ω)
		
		M_ωℓ = copy(M)

		M_ωℓ[diagind(M_ωℓ)[1:nr-2]] .+= @. -ω^2 * ρ[2:end-1]
		M_ωℓ[diagind(M_ωℓ)[nr+1:end]] .+= @. -ℓ*(ℓ+1)/(ω^2 * (ρ*r^2)[2:end-1])

		return M_ωℓ
	end

	function compute_radial_component_of_Gfn_onemode(ω,ℓ;r_src=r_src_default,tangential=false)

		# Solar model

		Sr,Sh = source(ω,ℓ,r_src=r_src)

		M = ℒωℓr(ω,ℓ);

		H = M\Sr;

		αrℓω = H[1:nr-1]
		prepend!(αrℓω,0)

		βrℓω = H[nr:end]
		append!(βrℓω,0)

		if tangential

			H = M\Sh;

			αhℓω = H[1:nr-1]
			prepend!(αhℓω,0)

			βhℓω = H[nr:end]
			append!(βhℓω,0)
		else
			αhℓω = βhℓω = zeros(nr)
		end

		return αrℓω,βrℓω,αhℓω,βhℓω
	end

	function Gfn_path_from_source_radius(r_src::Real)
		user=ENV["USER"]
		scratch=get(ENV,"SCRATCH","/scratch/$user")
		return "$scratch/Greenfn_src$((r_src/Rsun > 0.99 ? 
										(@sprintf "%dkm" (Rsun-r_src)/1e5) : (@sprintf "%.2fRsun" r_src/Rsun) ))"
	end

	function frequency_grid(r_src=r_src_default;ν_low=2.0e-3,ν_high=4.5e-3,num_ν=1250,ν_Nyquist=16e-3)
		
		dν = (ν_high - ν_low)/(num_ν-1); dω = 2π*dν
		
		# choose values on a grid
		ν_low_index = Int64(floor(ν_low/dν)); ν_low = ν_low_index*dν
		ν_high_index = num_ν + ν_low_index - 1; ν_high = ν_high_index*dν;
		Nν_Gfn = ν_high_index - ν_low_index + 1
		ν_Nyquist_index = Int64(ceil(ν_Nyquist/dν)); ν_Nyquist = ν_Nyquist_index*dν
		
		Nν = ν_Nyquist_index + 1; Nt = 2*(Nν-1)
		ν_full = (0:ν_Nyquist_index).*dν;
		ν_arr = (ν_low_index:ν_high_index).*dν ;
		T=1/dν; dt = T/Nt;
		ν_start_zeros = ν_low_index # index starts from zero
		ν_end_zeros = ν_Nyquist_index - ν_high_index

		ω_arr = 2π .* ν_arr;

		Gfn_save_directory = Gfn_path_from_source_radius(r_src)
		if !isdir(Gfn_save_directory)
			mkdir(Gfn_save_directory)
		end
		@save(joinpath(Gfn_save_directory,"parameters.jld2"),
			ν_arr,ν_full,dν,dω,ν_start_zeros,ν_end_zeros,Nν,Nt,dt,T,Nν_Gfn,ν_Nyquist)
	end

	function append_parameters(r_src=r_src_default;kwargs...)
		Gfn_save_directory = Gfn_path_from_source_radius(r_src)
		paramfile = joinpath(Gfn_save_directory,"parameters.jld2")
		params = jldopen(paramfile,"a+")
		for (k,v) in Dict(kwargs)
			params[string(k)] = v
		end
		close(params)
	end

	function update_parameters(r_src=r_src_default;kwargs...)
		Gfn_save_directory = Gfn_path_from_source_radius(r_src)
		paramfile = joinpath(Gfn_save_directory,"parameters.jld2")
		params = load(paramfile)
		for (k,v) in Dict(kwargs)
			params[string(k)] = v
		end
		save(paramfile,params)
	end

	function all_components(r_src=r_src_default;ℓ_arr=1:100,kwargs...)

		Gfn_save_directory = Gfn_path_from_source_radius(r_src)

		if !isdir(Gfn_save_directory)
			mkdir(Gfn_save_directory)
		end

		println("Saving output to $Gfn_save_directory")

		frequency_grid(r_src;kwargs...);
		@load joinpath(Gfn_save_directory,"parameters.jld2") Nν_Gfn dω Nν_Gfn ν_arr ν_start_zeros

		println("$Nν_Gfn frequencies over $(@sprintf "%.1e" ν_arr[1]) to $(@sprintf "%.1e" ν_arr[end])")
		println("ℓ from $(ℓ_arr[1]) to $(ℓ_arr[end])")

		function compute_G_somemodes_serial_oneproc(rank)

			# Each processor will save all ℓ's  for a range of frequencies if number of frequencies can be split evenly across processors.
			ℓ_ω_proc = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,rank);
			
			save_path = joinpath(Gfn_save_directory,@sprintf "Gfn_proc_%03d.fits" rank)

			# file = jldopen(save_path,"w")
			file = FITS(save_path,"w")

			# save real and imaginary parts separately 
			G = zeros(nr,2,0:1,0:1,0:1,length(ℓ_ω_proc))

			T2 = (N2./g .+ ddr*log.(r.^2 .*ρ) )

			for (ind,(ℓ,ω_ind)) in enumerate(ℓ_ω_proc)
				
				ω = dω*(ω_ind + ν_start_zeros)
				αr,βr,αh,βh = compute_radial_component_of_Gfn_onemode(ω,ℓ);

				@. G[:,1,0,0,0,ind] = real(αr)
				@. G[:,2,0,0,0,ind] = imag(αr)
				@. G[:,1,1,0,0,ind] = real(Ω(ℓ,0) * βr/(ρ*r*ω^2))
				@. G[:,2,1,0,0,ind] = imag(Ω(ℓ,0) * βr/(ρ*r*ω^2))

				@. G[:,1,0,1,0,ind] = real(αh) / √2
				@. G[:,2,0,1,0,ind] = imag(αh) / √2
				
				G[:,1,1,1,0,ind] .=  real(r./√(ℓ*(ℓ+1)) .* (βh./(ρ.*c.^2) .+ T2.*αh + Dr*αh) ./2)
				G[:,2,1,1,0,ind] .=  imag(r./√(ℓ*(ℓ+1)) .* (βh./(ρ.*c.^2) .+ T2.*αh + Dr*αh) ./2)

				put!(tracker,0)
			end

			write(file,G.parent)
			close(file)
			return rank
		end

		w = workers_active(ℓ_arr,1:Nν_Gfn)
		num_procs = length(w)
		println("Number of workers: $num_procs")

		append_parameters(num_procs=num_procs,ℓ_arr=ℓ_arr)

		if isempty(w)
			return
		else
			tracker = RemoteChannel(()->Channel{Int64}(100),1)
			prog_bar = Progress(num_tasks, 1,"Green functions computed : ")
			@sync begin
				@async f = [@spawnat p compute_G_somemodes_serial_oneproc(rank) for (rank,p) in enumerate(w)]
				@async begin 
					
					for n in 1:num_tasks
						take!(tracker)
						next!(prog_bar)
					end
				end
			end
			close(tracker)
		end
		return nothing
	end

	function components_radial_source(r_src=r_src_default;ℓ_arr=1:100,kwargs...)

		Gfn_save_directory = Gfn_path_from_source_radius(r_src)

		if !isdir(Gfn_save_directory)
			mkdir(Gfn_save_directory)
		end

		println("Saving output to $Gfn_save_directory")

		frequency_grid(r_src;kwargs...);
		@load joinpath(Gfn_save_directory,"parameters.jld2") Nν_Gfn dω Nν_Gfn ν_arr ν_start_zeros

		println("$Nν_Gfn frequencies over $(@sprintf "%.1e" ν_arr[1]) to $(@sprintf "%.1e" ν_arr[end])")
		println("ℓ from $(ℓ_arr[1]) to $(ℓ_arr[end])")

		function compute_G_somemodes_serial_oneproc(rank)

			# Each processor will save all ℓ's  for a range of frequencies if number of frequencies can be split evenly across processors.
			ℓ_ω_proc = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,rank);
			
			# Gfn_arr = Vector{Gfn}(undef,length(ℓ_ω_proc));
			save_path = joinpath(Gfn_save_directory,@sprintf "Gfn_proc_%03d.fits" rank)

			# file = jldopen(save_path,"w")
			file = FITS(save_path,"w")

			# save real and imaginary parts separately 
			# indices are r,re-im,α,β,dorder,ℓω
			G = zeros(nr,2,0:1,0:0,0:1,length(ℓ_ω_proc))

			for (ind,(ℓ,ω_ind)) in enumerate(ℓ_ω_proc)
				
				ω = dω*(ω_ind + ν_start_zeros)
				# need only the radial-source components if line-of-sight is ignored
				αr,βr, = compute_radial_component_of_Gfn_onemode(ω,ℓ,tangential=false);

				# radial component for radial source
				@. G[:,1,0,0,0,ind] = real(αr)
				@. G[:,2,0,0,0,ind] = imag(αr)
				G[:,1,0,0,1,ind] .= ddr*G[:,1,0,0,0,ind]
				G[:,2,0,0,1,ind] .= ddr*G[:,2,0,0,0,ind]
				# tangential component for radial source
				@. G[:,1,1,0,0,ind] = real(Ω(ℓ,0) * βr/(ρ*r*ω^2))
				@. G[:,2,1,0,0,ind] = imag(Ω(ℓ,0) * βr/(ρ*r*ω^2))
				G[:,1,1,0,1,ind] .= ddr*G[:,1,1,0,0,ind]
				G[:,2,1,0,1,ind] .= ddr*G[:,2,1,0,0,ind]

				put!(tracker,0)
			end

			write(file,G.parent)

			close(file)
			return rank
		end

		w = workers_active(ℓ_arr,1:Nν_Gfn)
		num_procs = length(w)
		println("Number of workers: $num_procs")
		num_tasks = length(ℓ_arr)*Nν_Gfn

		append_parameters(num_procs=num_procs,ℓ_arr=ℓ_arr)

		if isempty(w)
			return
		else
			tracker = RemoteChannel(()->Channel{Int64}(100),1)
			prog_bar = Progress(num_tasks, 1,"Green functions computed : ")
			@sync begin
				@async f = [@spawnat p compute_G_somemodes_serial_oneproc(rank) for (rank,p) in enumerate(w)]
				@async begin 
					
					for n in 1:num_tasks
						take!(tracker)
						next!(prog_bar)
					end
				end
			end
			close(tracker)
		end
		return nothing
	end

end

################################################################
# Three dimensional Green function
################################################################

# module Greenfn_3D

# 	using Reexport,FFTW,FastGaussQuadrature
# 	using OffsetArrays, FITSIO,Printf,JLD2,ParallelUtilities
# 	@reexport using Main.Greenfn_radial
# 	import Main.Greenfn_radial: Gfn_path_from_source_radius
# 	@reexport using Legendre,PointsOnASphere,TwoPointFunctions,VectorFieldsOnASphere

# 	Gfn_path_from_source_radius(x::Point3D) = Gfn_path_from_source_radius(x.r)

# 	function Powspec(ω)
# 		σ = 2π*0.4e-3
# 		ω0 = 2π*3e-3
# 		exp(-(ω-ω0)^2/(2σ^2))
# 	end

# 	function load_Greenfn_radial_coordinates_onefreq(Gfn_save_directory,ν,ν_arr,ℓ_arr,num_procs)

#     	ν_test_index = argmin(abs.(ν_arr .- ν));

# 		proc_id_start = get_processor_id_from_split_array(ℓ_arr,axes(ν_arr,1),(ℓ_arr[1],ν_test_index),num_procs);
# 	    proc_id_end = get_processor_id_from_split_array(ℓ_arr,axes(ν_arr,1),(ℓ_arr[end],ν_test_index),num_procs);

# 	    Gfn_arr_onefreq = Vector{Gfn}(undef,length(ℓ_arr))

# 	    Gfn_ℓ_index = 1
# 	    for proc_id in proc_id_start:proc_id_end
# 	    	G_proc_file = joinpath(Gfn_save_directory,@sprintf "Gfn_proc_%03d.jld2" proc_id)
# 	    	@load G_proc_file Gfn_arr
# 	    	for G in Gfn_arr
	    		
# 	    		if G.mode.ω_ind == ν_test_index
# 		    		Gfn_arr_onefreq[Gfn_ℓ_index] = G
# 		    		Gfn_ℓ_index += 1
# 	    		end

# 	    		if Gfn_ℓ_index > length(ℓ_arr)
# 	    			break
# 	    		end
# 	    	end
# 	    end
# 	    return Gfn_arr_onefreq
#     end

#     function compute_3D_Greenfn_helicity_angular_sections_onefreq(x′,(nθ,nϕ)::NTuple{2,Int64},ν::Real=3e-3,procid=myid()-1)

# 		Gfn_save_directory = Gfn_path_from_source_radius(x′)

#     	@load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs
    	
#     	Gfn_radial_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_save_directory,ν,ν_arr,ℓ_arr,num_procs)

#     	ℓmax = ℓ_arr[end]

#     	θ_full = LinRange(0,π,nθ) # might need non-uniform grid in θ    	
#     	ϕ_full = LinRange(0,2π,nϕ)

#     	θ_ϕ_iterator = split_product_across_processors(θ_full,ϕ_full,nworkers(),procid)

#     	function compute_3D_Greenfn_helicity_angular_sections_onefreq(x′,Gfn_radial_arr,θ_ϕ_iterator,ℓmax::Integer,ν::Real=3e-3,procid=myid()-1)

# 	    	Gfn_3D_arr = zeros(ComplexF64,1:nr,1:length(θ_ϕ_iterator),-1:1)

# 	    	d01Pl_cosχ = OffsetArray{Float64}(undef,0:ℓmax,0:1)
# 	    	Pl_cosχ = view(d01Pl_cosχ,:,0)
# 	    	dPl_cosχ = view(d01Pl_cosχ,:,1)

# 	    	for (θϕ_ind,(θ,ϕ)) in enumerate(θ_ϕ_iterator)
	    		
# 	    		Pl_dPl!( d01Pl_cosχ, cosχ((θ,ϕ),x′) )
		    	
# 		    	for Gfn in Gfn_radial_arr

# 		    		ℓ = Gfn.mode.ℓ 

# 		    		(ℓ<1) && continue

# 		    		G00 = view(Gfn.G,:,1)
# 		    		G10 = view(Gfn.G,:,2)

# 		    		# (-1,0) component
# 		    		em1_dot_∇_n_dot_n′ = 1/√2 * (∂θ₁cosχ((θ,ϕ),x′) + im * ∇ϕ₁cosχ((θ,ϕ),x′) )	    		
# 		    		@. Gfn_3D_arr[1:nr,θϕ_ind,-1] += (2ℓ +1)/4π  * G10/Ω(ℓ,0) * dPl_cosχ[ℓ] * em1_dot_∇_n_dot_n′

# 		    		# (0,0) component
# 		    		@. Gfn_3D_arr[1:nr,θϕ_ind,0] +=  (2ℓ +1)/4π * G00 * Pl_cosχ[ℓ]
		    		
# 		    		# (1,0) component
# 		    		e1_dot_∇_n_dot_n′ = 1/√2 * (-∂θ₁cosχ((θ,ϕ),x′) + im * ∇ϕ₁cosχ((θ,ϕ),x′) )
# 		    		@. Gfn_3D_arr[1:nr,θϕ_ind,1] +=  (2ℓ +1)/4π  * G10/Ω(ℓ,0) * dPl_cosχ[ℓ] * e1_dot_∇_n_dot_n′

# 		    	end
# 		    end

# 		    return Gfn_3D_arr
# 	    end

# 	    compute_3D_Greenfn_helicity_angular_sections_onefreq(x′,Gfn_radial_arr,θ_ϕ_iterator,ℓmax,ν,procid)
#     end

#     function compute_3D_Greenfn_spherical_angular_sections_onefreq(x′,
#     	θ_full::AbstractArray,ϕ_full::AbstractArray,ν::Real=3e-3,procid=myid()-1)

# 		Gfn_save_directory = Gfn_path_from_source_radius(x′)

#     	@load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs
    	
#     	Gfn_radial_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_save_directory,ν,ν_arr,ℓ_arr,num_procs)

#     	ℓmax = ℓ_arr[end]

#     	θ_ϕ_iterator = split_product_across_processors(θ_full,ϕ_full,nworkers(),procid)

#     	function compute_3D_Greenfn_spherical_angular_sections_onefreq(x′,Gfn_radial_arr,θ_ϕ_iterator,ℓmax::Integer,ν::Real=3e-3,procid=myid()-1)

# 	    	Gfn_3D_arr = zeros(ComplexF64,nr,length(θ_ϕ_iterator),3)

# 	    	d01Pl_cosχ = OffsetArray{Float64}(undef,0:ℓmax,0:1)
# 	    	Pl_cosχ = view(d01Pl_cosχ,:,0)
# 	    	dPl_cosχ = view(d01Pl_cosχ,:,1)

# 	    	for (θϕ_ind,(θ,ϕ)) in enumerate(θ_ϕ_iterator)
	    		
# 	    		Pl_dPl!( d01Pl_cosχ, cosχ((θ,ϕ),x′) )
		    	
# 		    	for Gfn in Gfn_radial_arr
		    		
# 		    		ℓ = Gfn.mode.ℓ 

# 		    		(ℓ<1) && continue

# 		    		G00 = view(Gfn.G,:,1)
# 		    		G10 = view(Gfn.G,:,2)

# 		    		# (r,r) component
# 		    		@. Gfn_3D_arr[1:nr,θϕ_ind,1] +=  (2ℓ+1)/4π * G00 * Pl_cosχ[ℓ]

# 		    		# (θ,r) component
# 		    		eθ_dot_∇_n_dot_n′ = ∂θ₁cosχ((θ,ϕ),x′)
# 		    		@. Gfn_3D_arr[1:nr,θϕ_ind,2] += (2ℓ+1)/4π * G10/Ω(ℓ,0) * dPl_cosχ[ℓ] * eθ_dot_∇_n_dot_n′
		    		
# 		    		# (ϕ,r) component
# 		    		eϕ_dot_∇_n_dot_n′ = ∇ϕ₁cosχ((θ,ϕ),x′)
# 		    		@. Gfn_3D_arr[1:nr,θϕ_ind,3] +=  (2ℓ+1)/4π * G10/Ω(ℓ,0) * dPl_cosχ[ℓ] * eϕ_dot_∇_n_dot_n′

# 		    	end
# 		    end

# 		    return Gfn_3D_arr
# 	    end

# 	    compute_3D_Greenfn_spherical_angular_sections_onefreq(x′,Gfn_radial_arr,θ_ϕ_iterator,ℓmax,ν,procid) 
#     end

#     function uϕ∇ϕG_uniform_rotation_angular_sections_onefreq(x′,
#     	θ_full::AbstractArray,ϕ_full::AbstractArray,ν::Real=3e-3,procid=myid()-1)

# 		Gfn_save_directory = Gfn_path_from_source_radius(x′)

#     	@load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs
    	
#     	Gfn_radial_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_save_directory,ν,ν_arr,ℓ_arr,num_procs)

#     	ℓmax = ℓ_arr[end]

#     	θ_ϕ_iterator = split_product_across_processors(θ_full,ϕ_full,nworkers(),procid)

#     	uϕ∇ϕG = zeros(ComplexF64,nr,length(θ_ϕ_iterator),3)

#     	d02Pl_cosχ = OffsetArray{Float64}(undef,0:ℓmax,0:2)
#     	Pl = view(d02Pl_cosχ,:,0)
#     	dPl = view(d02Pl_cosχ,:,1)
#     	d2Pl = view(d02Pl_cosχ,:,2)

#     	for (θϕ_ind,(θ,ϕ)) in enumerate(θ_ϕ_iterator)
    		
#     		Legendre.Pl_dPl_d2Pl!( d02Pl_cosχ, cosχ((θ,ϕ),x′) )
	    	
# 	    	for Gfn in Gfn_radial_arr
	    		
# 	    		ℓ = Gfn.mode.ℓ 

# 	    		(ℓ<1) && continue

# 	    		G00 = view(Gfn.G,:,1)
# 	    		G10 = view(Gfn.G,:,2)

# 	    		# (r,r) component
# 	    		∂ϕPl = dPl[ℓ] * ∂ϕ₁cosχ((θ,ϕ),x′)
# 	    		@. uϕ∇ϕG[:,θϕ_ind,1] +=  (2ℓ+1)/4π * (G00-G10/Ω(ℓ,0)) * ∂ϕPl
# 	    		# @. uϕ∇ϕG[:,θϕ_ind,1] +=  (2ℓ+1)/4π * G00 * ∂ϕPl
	    		
# 	    		# (θ,r) component
# 	    		d2Pl_∂θcosχ_∂ϕcosχ = d2Pl[ℓ] * ∂ϕ₁cosχ((θ,ϕ),x′) * ∂θ₁cosχ((θ,ϕ),x′)
# 	    		@. uϕ∇ϕG[:,θϕ_ind,2] += (2ℓ+1)/4π  * G10/Ω(ℓ,0) * d2Pl_∂θcosχ_∂ϕcosχ

# 	    		# (ϕ,r) component
# 	    		dϕ∇ϕPl = d2Pl[ℓ] * ∇ϕ₁cosχ((θ,ϕ),x′) * ∂ϕ₁cosχ((θ,ϕ),x′) + dPl[ℓ] * ∇ϕ₁∂ϕ₁cosχ((θ,ϕ),x′)
# 	    		∂θPl =  dPl[ℓ] * ∂θ₁cosχ((θ,ϕ),x′)
# 	    		@. uϕ∇ϕG[:,θϕ_ind,3] +=  (2ℓ+1)/4π  * (G10/Ω(ℓ,0) * (dϕ∇ϕPl + cos(θ) * ∂θPl ) + G00 * sin(θ) * Pl[ℓ])
# 	    		# @. uϕ∇ϕG[:,θϕ_ind,3] +=  (2ℓ+1)/4π  * (G00 * sin(θ) * Pl[ℓ])

# 	    	end
# 	    end

# 	    return uϕ∇ϕG
#     end

#     function δC_uniform_rotation_helicity_angular_sections_onefreq(x1,x2,nθ,nϕ,ν=3e-3,procid=myid()-1)

#     	Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

# 		@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs
# 		ℓmax = ℓ_arr[end]

# 		θ_full = LinRange(0,π,nθ)
# 		ϕ_full = LinRange(0,2π,nϕ)
# 		θ_ϕ_iterator = split_product_across_processors(θ_full,ϕ_full,nworkers(),procid)

# 		ν_ind = argmin(abs.(ν_arr .- ν))
# 		ω = 2π*ν_arr[ν_ind]

# 		Gfn_radial_arr_onefreq = load_Greenfn_radial_coordinates_onefreq(Gfn_directory_x1,ν,ν_arr,ℓ_arr,num_procs)

# 		Gfn3D_x1 = compute_3D_Greenfn_helicity_angular_sections_onefreq(x1,Gfn_radial_arr_onefreq,θ_ϕ_iterator,ℓmax,ν,procid)
# 		Gfn3D_x2 = compute_3D_Greenfn_helicity_angular_sections_onefreq(x2,Gfn_radial_arr_onefreq,θ_ϕ_iterator,ℓmax,ν,procid)

# 	    Yℓ1 = OffsetArray{ComplexF64}(undef,0:ℓmax,-2:2)
# 	    Yℓ2 = OffsetArray{ComplexF64}(undef,0:ℓmax,-2:2)

# 	    r₁_ind = argmin(abs.(r .- x1.r))
# 	    r₂_ind = argmin(abs.(r .- x2.r))

# 		integrand = zeros(ComplexF64,nr,length(θ_ϕ_iterator))

# 		for (θϕ_ind,(θ,ϕ)) in enumerate(θ_ϕ_iterator)

# 			ϕ₁,θ₁,_ = add_rotations_euler_angles((-x1.ϕ,-x1.θ,0),(0,θ,ϕ))
# 			ϕ₂,θ₂,_ = add_rotations_euler_angles((-x2.ϕ,-x2.θ,0),(0,θ,ϕ))

# 			Ylm!(Yℓ1,θ₁,ϕ₁)
# 			Ylm!(Yℓ2,θ₂,ϕ₂)

# 				for Gfn in Gfn_radial_arr_onefreq
# 					ℓ = Gfn.mode.ℓ

# 					(ℓ < 2) && continue

# 					Gℓ = OffsetArray(Gfn.G,(0,-1))
# 					Gℓ[:,1] ./= Ω(ℓ,1)

# 					for α=-1:1
# 						@. integrand[1:nr,θϕ_ind] += sin(θ) * √((2ℓ+1)/4π) * (

# 						Gfn3D_x2[α,1:nr,θϕ_ind]*conj(Gfn.G[r₁_ind,1])*
# 						((Gfn.G[1:nr,abs(α)] * Ω(ℓ,α) - (α != -1 ? Gfn.G[1:nr,abs(α-1)] : 0) )*Yℓ1[ℓ,α-1] + 
# 						(Gfn.G[1:nr,abs(α)] * Ω(ℓ,-α) - (α != 1 ? Gfn.G[1:nr,abs(α+1)] : 0) )*Yℓ1[ℓ,α+1]) +
						
# 						conj(Gfn3D_x1[α,1:nr,θϕ_ind])*Gfn.G[r₂_ind,1]*
# 						((conj(Gfn.G[1:nr,abs(α)]) * Ω(ℓ,-α) - (α != 1 ? conj(Gfn.G[1:nr,abs(α+1)]) : 0) )*conj(Yℓ2[ℓ,α+1]) + 
# 						(conj(Gfn.G[1:nr,abs(α)]) * Ω(ℓ,α) - (α != -1 ? conj(Gfn.G[1:nr,abs(α-1)]) : 0) )*conj(Yℓ2[ℓ,α-1]) )

# 						)

# 					end

# 			end
# 		end

# 		return integrate.simps(ρ .* r.^2 .* integrand,x=r)
#     end

#     function δC_uniform_rotation_spherical_angular_sections_onefreq(x1,x2,θ_full,ϕ_full,ν=3e-3,procid=myid()-1)

#     	Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

# 		@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs
# 		ℓmax = ℓ_arr[end]

# 		P1 = OffsetArray{Float64}(undef,0:ℓmax,0:2)
# 		P2 = OffsetArray{Float64}(undef,0:ℓmax,0:2)

# 		∂ϕPl1 = OffsetArray{Float64}(undef,0:ℓmax)
# 		∂ϕPl2 = OffsetArray{Float64}(undef,0:ℓmax)
# 		∇ϕPl1 = OffsetArray{Float64}(undef,0:ℓmax)
# 		∇ϕPl2 = OffsetArray{Float64}(undef,0:ℓmax)
# 		∂θPl1 = OffsetArray{Float64}(undef,0:ℓmax)
# 		∂θPl2 = OffsetArray{Float64}(undef,0:ℓmax)
# 		∂ϕ∂θPl1 = OffsetArray{Float64}(undef,0:ℓmax)
# 		∂ϕ∂θPl2 = OffsetArray{Float64}(undef,0:ℓmax)
# 		∂²ϕPl1 = OffsetArray{Float64}(undef,0:ℓmax)
# 		∂²ϕPl2 = OffsetArray{Float64}(undef,0:ℓmax)
# 		∇ϕ∂ϕPl1 = OffsetArray{Float64}(undef,0:ℓmax)
# 		∇ϕ∂ϕPl2 = OffsetArray{Float64}(undef,0:ℓmax)

# 		θ_ϕ_iterator = split_product_across_processors(θ_full,ϕ_full,nworkers(),procid)

# 		ν_ind = argmin(abs.(ν_arr .- ν))
# 		ω = 2π*ν_arr[ν_ind]

# 		Gfn_radial_arr_onefreq = load_Greenfn_radial_coordinates_onefreq(Gfn_directory_x1,ν,ν_arr,ℓ_arr,num_procs)

# 		Gfn3D_x1 = compute_3D_Greenfn_spherical_angular_sections_onefreq(x1,Gfn_radial_arr_onefreq,θ_ϕ_iterator,ℓmax,ν,procid)
# 		Gfn3D_x2 = compute_3D_Greenfn_spherical_angular_sections_onefreq(x2,Gfn_radial_arr_onefreq,θ_ϕ_iterator,ℓmax,ν,procid)

# 	    r₁_ind = argmin(abs.(r .- x1.r))
# 	    r₂_ind = argmin(abs.(r .- x2.r))

# 		integrand = zeros(ComplexF64,nr,length(θ_ϕ_iterator))

# 		for (θϕ_ind,(θ,ϕ)) in enumerate(θ_ϕ_iterator)

# 			Legendre.Pl_dPl_d2Pl!(P1,cosχ((θ,ϕ),x1),ℓmax=ℓmax)
# 			Pl1 = view(P1,:,0)
# 			dPl1 = view(P1,:,1)
# 			d2Pl1 = view(P1,:,2)

# 			∂θPl1 .= dPl1 .* ∂θ₁cosχ((θ,ϕ),x1)
# 			∂ϕPl1 .= dPl1 .* ∂ϕ₁cosχ((θ,ϕ),x1)
# 			∇ϕPl1 .= dPl1 .* ∇ϕ₁cosχ((θ,ϕ),x1)
# 			∂ϕ∂θPl1 .= d2Pl1 .* ∂ϕ₁cosχ((θ,ϕ),x1) .* ∂θ₁cosχ((θ,ϕ),x1) .+ dPl1 .* ∂θ₁∂ϕ₁cosχ((θ,ϕ),x1)
# 			∂²ϕPl1 .= d2Pl1 .* ∂ϕ₁cosχ((θ,ϕ),x1)^2 .+ dPl1 .* ∂²ϕ₁cosχ((θ,ϕ),x1)
# 			∇ϕ∂ϕPl1 .= d2Pl1 .* ∇ϕ₁cosχ((θ,ϕ),x1) .* ∂ϕ₁cosχ((θ,ϕ),x1) .+ dPl1 .* ∇ϕ₁∂ϕ₁cosχ((θ,ϕ),x1)

# 			Legendre.Pl_dPl_d2Pl!(P2,cosχ((θ,ϕ),x2),ℓmax=ℓmax)
# 			Pl2 = view(P2,:,0)
# 			dPl2 = view(P2,:,1)
# 			d2Pl2 = view(P2,:,2)

# 			∂θPl2 .= dPl2 .* ∂θ₁cosχ((θ,ϕ),x2)
# 			∂ϕPl2 .= dPl2 .* ∂ϕ₁cosχ((θ,ϕ),x2)
# 			∇ϕPl2 .= dPl2 .* ∇ϕ₁cosχ((θ,ϕ),x2)
# 			∂ϕ∂θPl2 .= d2Pl2 .* ∂ϕ₁cosχ((θ,ϕ),x2) .* ∂θ₁cosχ((θ,ϕ),x2) .+ dPl2 .* ∂θ₁∂ϕ₁cosχ((θ,ϕ),x2)
# 			∂²ϕPl2 .= d2Pl2 .* ∂ϕ₁cosχ((θ,ϕ),x2)^2 .+ dPl2 .* ∂²ϕ₁cosχ((θ,ϕ),x2)
# 			∇ϕ∂ϕPl2 .= d2Pl2 .* ∇ϕ₁cosχ((θ,ϕ),x2) .* ∂ϕ₁cosχ((θ,ϕ),x2) .+ dPl2 .* ∇ϕ₁∂ϕ₁cosχ((θ,ϕ),x2)


# 			for Gfn in Gfn_radial_arr_onefreq
# 				ℓ = Gfn.mode.ℓ

# 				(ℓ < 1) && continue

# 				G00 = view(Gfn.G,:,1)
# 				G10 = view(Gfn.G,:,2)

# 				@. integrand[1:nr,θϕ_ind] += (2ℓ+1)/4π * (

# 				- G00[r₂_ind]*(
# 					conj(Gfn3D_x1[:,θϕ_ind,1])*conj(G00 - G10/Ω(ℓ,0)) * ∂ϕPl2[ℓ]  + 
# 					conj(Gfn3D_x1[:,θϕ_ind,2])*conj(G10/Ω(ℓ,0)) * (∂ϕ∂θPl2[ℓ] - cos(θ)*∇ϕPl2[ℓ]) +
# 					conj(Gfn3D_x1[:,θϕ_ind,3])*(conj(G10/Ω(ℓ,0))* (∇ϕ∂ϕPl2[ℓ] + cos(θ) * ∂θPl2[ℓ]) + conj(G00)*sin(θ)*Pl2[ℓ])
# 					) +

# 				conj(G00[r₁_ind])*(
# 					Gfn3D_x2[:,θϕ_ind,1]*(G00 - G10/Ω(ℓ,0)) * ∂ϕPl1[ℓ]  + 
# 					Gfn3D_x2[:,θϕ_ind,2]*G10/Ω(ℓ,0) * (∂ϕ∂θPl1[ℓ] - cos(θ)*∇ϕPl1[ℓ]) +
# 					Gfn3D_x2[:,θϕ_ind,3]*(G10/Ω(ℓ,0)* (∇ϕ∂ϕPl1[ℓ] + cos(θ) * ∂θPl1[ℓ]) + G00*sin(θ)*Pl1[ℓ] ) 
# 					)
# 				)
			
# 			end

# 		end

# 		return integrate.simps(ρ .* r.^2 .* integrand,x=r)
#     end

# 	function compute_3D_Greenfn_components(x′::Point3D,θ::AbstractArray,ϕ::AbstractArray,ν::Real=3e-3;basis="spherical")
		
# 		nθ = length(θ); 
# 		nϕ=length(ϕ);
		
# 		f = eval(Symbol("compute_3D_Greenfn_$(basis)_angular_sections_onefreq"))

# 		Gfn_3D_futures_procs_used = [@spawnat p f(x′,θ,ϕ,ν) for p in workers_active(1:nθ,1:nϕ)]

# 		Gfn_3D = reshape(cat(fetch.(Gfn_3D_futures_procs_used)...,dims=2),nr,nθ,nϕ,3)
# 	end

# 	function compute_u_dot_∇_G_uniform_rotation(x′::Point3D,θ::AbstractArray,ϕ::AbstractArray,ν::Real=3e-3)
# 		Gfn_save_directory = Gfn_path_from_source_radius(x′)

# 		# @load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs

# 		# ℓmax = ℓ_arr[end]
		
# 		nθ = length(θ); 
# 		nϕ=length(ϕ);
		
# 		Gfn_3D_futures_procs_used = [@spawnat p uϕ∇ϕG_uniform_rotation_angular_sections_onefreq(x′,θ,ϕ,ν) for p in workers_active(1:nθ,1:nϕ)]

# 		Gfn_3D = reshape(cat(fetch.(Gfn_3D_futures_procs_used)...,dims=2),nr,nθ,nϕ,3)
# 	end

# 	function δLG_uniform_rotation(x_src::Point3D,θ::AbstractArray,ϕ::AbstractArray,ν::Real=3e-3)
		
# 		Gfn_save_directory = Gfn_path_from_source_radius(x_src)
# 		@load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs

# 		ν = ν_arr[argmin(abs.(ν_arr .- ν))]; ω = 2π*ν

# 		udot∇G = compute_u_dot_∇_G_uniform_rotation(x_src,θ,ϕ,ν)

# 		return @. -2im*ω*ρ*udot∇G
# 	end

# 	function δGrr_uniform_rotation_firstborn(x1 = Point3D(Rsun-75e5,π/2,0),x_src=Point3D(Rsun-75e5,π/2,π/3),
# 		ν::Real=3e-3)
# 		# δG_ik(x1,x2) = -∫dx Gij(x1,x)[δL(x)G(x,x2)]jk = -∫dx Gji(x,x1)[δL(x)G(x,x2)]jk
# 		# We compute δG_rr(x1,x2) = -∫dx Gjr(x,x1)[δL(x)G(x,x2)]jr

# 		Gfn_path = Gfn_path_from_source_radius(x1)
# 		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs

# 		ν_on_grid = ν_arr[argmin(abs.(ν_arr .- ν))]

# 		@printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_arr[1] ℓ_arr[end]
# 		ℓmax = ℓ_arr[end]
		
# 		# θ = LinRange(0,π,2ℓmax)[2:end-1]; 
# 		cosθGL,wGL = gausslegendre(ℓmax); θ=acos.(cosθGL);
# 		nθ = length(θ); dθ = θ[2] - θ[1];
# 		ϕ = LinRange(0,2π,2ℓmax); dϕ = ϕ[2] - ϕ[1]; nϕ=length(ϕ)
# 		θ3D = reshape(θ,1,nθ,1)

# 		δLG = δLG_uniform_rotation(x_src,θ,ϕ,ν_on_grid)
# 		G = compute_3D_Greenfn_components(x1,θ,ϕ,ν_on_grid,basis="spherical")
# 		integrand = dropdims(sum(G .* δLG, dims=4),dims=4)

# 		dG = - integrate.simps(dropdims(sum(wGL .* integrate.simps((@. r^2*integrand),x=r,axis=0),dims=1),dims=1),dx=dϕ)
# 	end

# 	function δGrr_uniform_rotation_firstborn_integrated_over_angle(x1= Point3D(Rsun-75e5,π/2,0),x_src=Point3D(Rsun-75e5,π/2,π/3),
# 		ν::Real=3e-3)
# 		# δG_ik(x1,x2) = -∫dx Gij(x1,x)[δL(x)G(x,x2)]jk = -∫dx Gji(x,x1)[δL(x)G(x,x2)]jk
# 		# We compute δG_rr(x1,x2) = -∫dx Gjr(x,x1)[δL(x)G(x,x2)]jr

# 		Gfn_path = Gfn_path_from_source_radius(x1)
# 		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs

# 		ν_test_index = argmin(abs.(ν_arr .- ν))
# 		ν_on_grid = ν_arr[ν_test_index]
# 		ω = 2π * ν_on_grid

# 		@printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_arr[1] ℓ_arr[end]
# 		ℓmax = ℓ_arr[end]

# 		Gfn_radial_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_path,ν,ν_arr,ℓ_arr,num_procs)

# 		dPl_cosχ = dPl(cosχ(x1,x_src),ℓmax=ℓ_arr[end])

# 		Gfn_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_path,ν,ν_arr,ℓ_arr,num_procs)

# 		δG = zeros(ComplexF64,nr)
	    	
# 		for Gfn in Gfn_arr
# 			ℓ = Gfn.mode.ℓ
# 			ω_ind = Gfn.mode.ω_ind

# 		    G0 = view(Gfn.G,:,1)
# 		    G1 = view(Gfn.G,:,2)

# 		    ((ℓ<1) || (ω_ind != ν_test_index )) && continue

# 		    δG .+= (2ℓ+1)/4π .* (@. G0^2 - 2G0*G1/Ω(ℓ,0) + (ℓ*(ℓ+1)-1)*(G1/Ω(ℓ,0))^2) .* (dPl_cosχ[ℓ]*∂ϕ₁cosχ(x1,x_src))
# 		end

# 		return 2im*ω*integrate.simps((@. r^2 * ρ * δG),x=r)
# 	end

# 	function δGrr_uniform_rotation_rotatedwaves(x1::Point3D,x_src::Point3D,ν=3e-3)
		
# 		# Assuming Ω_rotation = 1, scale the result up by Ω as needed
# 		Gfn_path = Gfn_path_from_source_radius(x1)
# 		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs

# 		ν_test_ind = argmin(abs.(ν_arr .- ν))

# 		ν_on_grid = ν_arr[ν_test_ind]

# 		@printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_arr[1] ℓ_arr[end]

# 		ν_ind_range = ν_test_ind-2:ν_test_ind+2
# 		ω_arr = 2π .* ν_arr[ν_ind_range]
# 		dω = ω_arr[2] - ω_arr[1]
		
# 		ℓmax = ℓ_arr[end]

# 		Grr_rsrc_equator = OffsetArray(zeros(ComplexF64,length(ν_ind_range)),ν_ind_range) # Green function at equator on the surface

# 		r₁_ind = argmin(abs.(r .- x1.r)) # observations at the source radius, can be anything as long as consistent

# 		∂ϕ₁Pl_cosχ = dPl(cosχ(x1,x_src),ℓmax=ℓmax) .* ∂ϕ₁cosχ(x1,x_src)

# 		ℓ_ωind_iter_on_proc = split_product_across_processors(ℓ_arr,ν_ind_range)

# 		proc_id_range = get_processor_range_from_split_array(ℓ_arr,axes(ν_arr,1),ℓ_ωind_iter_on_proc,num_procs);
		

# 		function summodes(proc_id,ν_ind_range,∂ϕ₁Pl_cosχ)

# 			G = OffsetArray(zeros(ComplexF64,length(ν_ind_range)),ν_ind_range)

# 			G_proc_file = joinpath(Gfn_path,@sprintf "Gfn_proc_%03d.jld2" proc_id)
# 	    	@load G_proc_file Gfn_arr
# 	    	for Gfn in Gfn_arr

# 	    		ℓ = Gfn.mode.ℓ
# 	    		ω_ind = Gfn.mode.ω_ind
			    
# 			    if (ω_ind ∉ ν_ind_range) || (ℓ<1)
# 			    	continue
# 			    end
	    	
# 	    		G0 = view(Gfn.G,:,1)
	    		
# 	    		G[ω_ind] += (2ℓ+1)/4π * G0[r₁_ind] * ∂ϕ₁Pl_cosχ[ℓ]
	    		
# 	    	end

# 			return G
# 		end

# 		Grr_rsrc_equator = @distributed (+) for proc_id in proc_id_start:proc_id_end
# 								summodes(proc_id,ν_ind_range,∂ϕ₁Pl_cosχ)
# 							end

# 	    D_op = dbydr(ω_arr,Dict(2=>3)) 
# 	    # display(Matrix(D_op))

# 	    ∂ωG = D_op*Grr_rsrc_equator.parent
# 		return -im*∂ωG[div(length(ν_ind_range),2) + 1]
# 	end

# 	function uϕ_dot_∇G_finite_difference(x::Point3D,θ::AbstractArray,ϕ::AbstractArray,ν::Real=3e-3)
		
# 		dϕ = 1e-8; dϕ_grid = ϕ[2] - ϕ[1]
# 		θ3D = reshape(θ,1,length(θ),1)
# 		G = compute_3D_Greenfn_components(x,θ,ϕ,ν,basis="spherical")
		
# 		G_plus_r = compute_3D_Greenfn_components(Point3D(x.r,x.θ,x.ϕ-dϕ),θ,ϕ,ν,basis="spherical")[:,:,:,1] # rotate source in opposite direction
# 		G_minus_r = compute_3D_Greenfn_components(Point3D(x.r,x.θ,x.ϕ+dϕ),θ,ϕ,ν,basis="spherical")[:,:,:,1] # rotate source in opposite direction

# 		G_r = view(G,:,:,:,1)
# 		G_θ = view(G,:,:,:,2)
# 		G_ϕ = view(G,:,:,:,3)

# 		uϕ∇ϕG_r = @. (G_plus_r - G_minus_r)/2dϕ- G_ϕ * sin(θ3D)
# 		uϕ∇ϕG_θ = (-roll(G_θ,-2) .+ 8roll(G_θ,-1) .- 8roll(G_θ,1) .+ roll(G_θ,2))./12dϕ_grid .- G_ϕ .* cos.(θ3D)
# 		uϕ∇ϕG_ϕ = (-roll(G_ϕ,-2) .+ 8roll(G_ϕ,-1) .- 8roll(G_ϕ,1) .+ roll(G_ϕ,2))./12dϕ_grid .+ G_θ .* cos.(θ3D) .+ G_r .* sin.(θ3D)
		
# 		return cat(uϕ∇ϕG_r,uϕ∇ϕG_θ,uϕ∇ϕG_ϕ,dims=4)
# 	end

# 	function δLG_uniform_rotation_finite_difference(x::Point3D,θ::AbstractArray,ϕ::AbstractArray,ν::Real=3e-3)
# 		# This is simply -2iωρ u⋅∇ G
# 		ω = 2π*ν
# 		return -2im .*ω.*ρ.*uϕ_dot_∇G_finite_difference(x,θ,ϕ,ν)
# 	end

# 	function δGrr_uniform_rotation_firstborn_finite_difference(x1 = Point3D(Rsun-75e5,π/2,0),x2=Point3D(Rsun-75e5,π/2,π/3),ν=3e-3)
# 		# δG_ik(x1,x2) = -∫dx Gij(x1,x)[δL(x)G(x,x2)]jk = -∫dx Gji(x,x1)[δL(x)G(x,x2)]jk
# 		# We compute δG_rr(x1,x2) = -∫dx Gjr(x,x1)[δL(x)G(x,x2)]jr

# 		Gfn_path = Gfn_path_from_source_radius(x1)
# 		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs
# 		ν = ν_arr[argmin(abs.(ν_arr .- ν))]
# 		ℓmax = ℓ_arr[end]

# 		θ = LinRange(0,π,2ℓmax)[2:end-1]; 
# 		# cosθGL,wGL = gausslegendre(2ℓmax); θ=acos.(cosθGL);
# 		nθ = length(θ); dθ = θ[2] - θ[1];
# 		ϕ = LinRange(0,2π,8ℓmax); dϕ = ϕ[2] - ϕ[1];

# 		G = compute_3D_Greenfn_components(x1,θ,ϕ,ν,basis="spherical");
# 		δLG = δLG_uniform_rotation_finite_difference(x2,θ,ϕ,ν);
		
# 		integrand = dropdims(sum(G .* δLG, dims=4),dims=4)
	
# 		dG = - integrate.simps(integrate.simps(sin.(θ).*integrate.simps((@. r^2*integrand),x=r,axis=0),dx=dθ,axis=0),dx=dϕ)
# 	end

# 	function δCω_uniform_rotation_firstborn(x1 = Point3D(Rsun-75e5,π/2,0),x2=Point3D(Rsun-75e5,π/2,π/3),ν=3e-3;Ω_rot = 20e2/Rsun)
		
# 		Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

# 		@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs

# 		ν_on_grid = ν_arr[argmin(abs.(ν_arr .- ν))]
# 		ω = 2π*ν_on_grid

# 		ℓmax = ℓ_arr[end]
# 		nθ,nϕ = 2ℓmax,2ℓmax

# 		θ = LinRange(0,π,nθ)
# 		ϕ = LinRange(0,2π,nϕ)

# 		workers_active = [i for i in workers() if length(split_product_across_processors(1:nθ,1:nϕ,nworkers(),i-1))!=0]

# 		dC_Ω_futures = [@spawnat p δC_uniform_rotation_helicity_angular_sections_onefreq(x1,x2,nθ,nϕ,ν) for p in workers_active]

# 		dC_Ω = reshape(vcat(fetch.(dC_Ω_futures)...),nθ,nϕ)

# 		dC = -√2 *Ω_rot* ω^3 * Powspec(ω) * ∮dΩ(dC_Ω,θ,ϕ)
# 	end

# 	function δCω_uniform_rotation_firstborn_krishnendu(x1 = Point3D(Rsun-75e5,π/2,0),x2=Point3D(Rsun-75e5,π/2,π/3),ν=3e-3;Ω_rot = 20e2/Rsun)
# 		Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

# 		@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs

# 		ν_on_grid = ν_arr[argmin(abs.(ν_arr .- ν))]
# 		ω = 2π*ν_on_grid

# 		@printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_arr[1] ℓ_arr[end]

# 		ℓmax = ℓ_arr[end]

# 		θ = LinRange(0,π,4ℓmax)[2:end-1]; nθ = length(θ)
# 		cosθGL,wGL = gausslegendre(4ℓmax); θGL=acos.(cosθGL); nθGL = length(θGL);
# 		ϕ = LinRange(0,2π,4ℓmax); dϕ = ϕ[2] - ϕ[1]; nϕ=length(ϕ)

# 		# linear θ grid
# 		workers_active = [i for i in workers() if length(split_product_across_processors(1:nθ,1:nϕ,nworkers(),i-1))!=0]
# 		dC_Ω_futures = [@spawnat p δC_uniform_rotation_spherical_angular_sections_onefreq(x1,x2,θ,ϕ,ν) for p in workers_active]
# 		dC_Ω = reshape(vcat(fetch.(dC_Ω_futures)...),nθ,nϕ)

# 		dC = 2im *Ω_rot* ω^3 * Powspec(ω) * integrate.simps(integrate.simps(dC_Ω.*sin.(θ),x=θ,axis=0),dx=dϕ)
# 		println("Linear θ grid: δC = $dC")

# 		# gauss-legendre nodes
# 		workers_active = [i for i in workers() if length(split_product_across_processors(1:nθGL,1:nϕ,nworkers(),i-1))!=0]
# 		dC_Ω_futures = [@spawnat p δC_uniform_rotation_spherical_angular_sections_onefreq(x1,x2,θGL,ϕ,ν) for p in workers_active]
# 		dC_Ω_GL = reshape(vcat(fetch.(dC_Ω_futures)...),nθGL,nϕ)

# 		dC = 2im *Ω_rot* ω^3 * Powspec(ω) * integrate.simps(dropdims(sum(wGL.*dC_Ω_GL,dims=1),dims=1),dx=dϕ)

# 		println("Gauss-Legendre quadrature: δC = $dC")
# 	end

# 	function δCω_uniform_rotation_firstborn(x1 = Point3D(Rsun-75e5,π/2,0),x2=Point3D(Rsun-75e5,π/2,π/3),ν=3e-3;Ω_rot = 20e2/Rsun)
		
# 		Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

# 		@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs

# 		ν_on_grid = ν_arr[argmin(abs.(ν_arr .- ν))]
# 		ω = 2π*ν_on_grid

# 		ℓmax = ℓ_arr[end]
# 		nθ,nϕ = 2ℓmax,2ℓmax

# 		θ = LinRange(0,π,nθ)
# 		ϕ = LinRange(0,2π,nϕ)

# 		workers_active = [i for i in workers() if length(split_product_across_processors(1:nθ,1:nϕ,nworkers(),i-1))!=0]

# 		dC_Ω_futures = [@spawnat p δC_uniform_rotation_helicity_angular_sections_onefreq(x1,x2,nθ,nϕ,ν) for p in workers_active]

# 		dC_Ω = reshape(vcat(fetch.(dC_Ω_futures)...),nθ,nϕ)

# 		dC = -√2 *Ω_rot* ω^3 * Powspec(ω) * ∮dΩ(dC_Ω,θ,ϕ)
# 	end

# 	function δCω_uniform_rotation_firstborn_krishnendu(x1 = Point3D(Rsun-75e5,π/2,0),x2=Point3D(Rsun-75e5,π/2,π/3),ν=3e-3;Ω_rot = 20e2/Rsun)
# 		Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

# 		@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs

# 		ν_on_grid = ν_arr[argmin(abs.(ν_arr .- ν))]
# 		ω = 2π*ν_on_grid

# 		@printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_arr[1] ℓ_arr[end]

# 		ℓmax = ℓ_arr[end]

# 		θ = LinRange(0,π,4ℓmax)[2:end-1]; nθ = length(θ)
# 		cosθGL,wGL = gausslegendre(4ℓmax); θGL=acos.(cosθGL); nθGL = length(θGL);
# 		ϕ = LinRange(0,2π,4ℓmax); dϕ = ϕ[2] - ϕ[1]; nϕ=length(ϕ)

# 		# linear θ grid
# 		workers_active = [i for i in workers() if length(split_product_across_processors(1:nθ,1:nϕ,nworkers(),i-1))!=0]
# 		dC_Ω_futures = [@spawnat p δC_uniform_rotation_spherical_angular_sections_onefreq(x1,x2,θ,ϕ,ν) for p in workers_active]
# 		dC_Ω = reshape(vcat(fetch.(dC_Ω_futures)...),nθ,nϕ)

# 		dC = 2im *Ω_rot* ω^3 * Powspec(ω) * integrate.simps(integrate.simps(dC_Ω.*sin.(θ),x=θ,axis=0),dx=dϕ)
# 		println("Linear θ grid: δC = $dC")

# 		# gauss-legendre nodes
# 		workers_active = [i for i in workers() if length(split_product_across_processors(1:nθGL,1:nϕ,nworkers(),i-1))!=0]
# 		dC_Ω_futures = [@spawnat p δC_uniform_rotation_spherical_angular_sections_onefreq(x1,x2,θGL,ϕ,ν) for p in workers_active]
# 		dC_Ω_GL = reshape(vcat(fetch.(dC_Ω_futures)...),nθGL,nϕ)

# 		dC = 2im *Ω_rot* ω^3 * Powspec(ω) * integrate.simps(dropdims(sum(wGL.*dC_Ω_GL,dims=1),dims=1),dx=dϕ)

# 		println("Gauss-Legendre quadrature: δC = $dC")
# 	end
# end