module parallel_utilities

	using Reexport
	@reexport using Distributed

	worker_rank() = myid()-minimum(workers())+1

	function split_across_processors(num_tasks::Integer,num_procs=nworkers(),proc_id=worker_rank())
		if num_procs == 1
			return arr₁
		end

		num_tasks_per_process,num_tasks_leftover = div(num_tasks,num_procs),mod(num_tasks,num_procs)

		num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
		task_start = num_tasks_per_process*(proc_id-1) + min(num_tasks_leftover+1,proc_id);

		return task_start:(task_start+num_tasks_on_proc-1)
	end

	function split_across_processors(arr₁,num_procs=nworkers(),proc_id=worker_rank())
		if isnothing(proc_id)
			return []
		end

		if num_procs == 1
			return arr₁
		end

		num_tasks = length(arr₁);

		num_tasks_per_process,num_tasks_leftover = div(num_tasks,num_procs),mod(num_tasks,num_procs)

		num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
		task_start = num_tasks_per_process*(proc_id-1) + min(num_tasks_leftover+1,proc_id);

		return Iterators.take(Iterators.drop(arr₁,task_start-1),num_tasks_on_proc)
	end

	function split_product_across_processors(arr₁,arr₂,num_procs=nworkers(),proc_id=worker_rank())

		# arr₁ will change faster
		return split_across_processors(Iterators.product(arr₁,arr₂),num_procs,proc_id)
	end

	function get_processor_id_from_split_array(arr₁,arr₂,(arr₁_value,arr₂_value)::Tuple,num_procs)
		# Find the closest match in arrays

		if (arr₁_value ∉ arr₁) || (arr₂_value ∉ arr₂)
			return nothing # invalid
		end
		
		num_tasks = length(arr₁)*length(arr₂);

		a1_match_index = argmin(abs.(arr₁ .- arr₁_value))
		a2_match_index = argmin(abs.(arr₂ .- arr₂_value))

		num_tasks_per_process,num_tasks_leftover = div(num_tasks,num_procs),mod(num_tasks,num_procs)

		proc_id = 1
		num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
		total_tasks_till_proc_id = num_tasks_on_proc

		task_no = 0

		for (ind2,a2) in enumerate(arr₂), (ind1,a1) in enumerate(arr₁)
			
			task_no +=1
			if task_no > total_tasks_till_proc_id
				proc_id += 1
				num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
				total_tasks_till_proc_id += num_tasks_on_proc
			end

			if ind2< a2_match_index
				continue
			end

			if (ind2 == a2_match_index) && (ind1 == a1_match_index)
				break
			end
		end

		return proc_id
	end

	function get_processor_range_from_split_array(arr₁,arr₂,modes_on_proc,num_procs)
		
		if isempty(modes_on_proc)
			return [] # empty range
		end

		tasks_arr = collect(modes_on_proc)
		proc_id_start = get_processor_id_from_split_array(arr₁,arr₂,first(tasks_arr),num_procs)
		proc_id_end = get_processor_id_from_split_array(arr₁,arr₂,last(tasks_arr),num_procs)
		return proc_id_start:proc_id_end
	end

	function get_index_in_split_array(modes_on_proc,(arr₁_value,arr₂_value))
		if isnothing(modes_on_proc)
			return nothing
		end
		for (ind,(t1,t2)) in enumerate(modes_on_proc)
			if (t1==arr₁_value) && (t2 == arr₂_value)
				return ind
			end
		end
		nothing
	end

	function procid_and_mode_index(ℓ_arr,ω_inds,(ℓ,ω_ind),num_procs)
		proc_id_mode = get_processor_id_from_split_array(ℓ_arr,ω_inds,(ℓ,ω_ind),num_procs)
		modes_in_procid_file = split_product_across_processors(ℓ_arr,ω_inds,num_procs,proc_id_mode)
		mode_index = get_index_in_split_array(modes_in_procid_file,(ℓ,ω_ind))
		return proc_id_mode,mode_index
	end

	workers_active(arr₁,arr₂) = [p for (rank,p) in enumerate(workers()) 
								if !isempty(split_product_across_processors(arr₁,arr₂,nworkers(),rank))]

	export split_product_across_processors,get_processor_id_from_split_array,get_processor_range_from_split_array,workers_active
	export get_index_in_split_array,procid_and_mode_index
end

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


module load_parameters

	using Reexport
	@reexport using Main.finite_difference,DelimitedFiles,Main.parallel_utilities,Polynomials

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

	export load_solar_model
	const Rsun,nr,r,dr,ddr,c,ρ,g,N2,γ_damping = load_solar_model()
	export Rsun,nr,r,dr,ddr,c,ρ,g,N2,γ_damping
end


#################################################################
# Green function radial components, main function
#################################################################

module Greenfn_radial

	using Reexport
	@reexport using Main.load_parameters,Main.parallel_utilities,Main.finite_difference
	using LinearAlgebra,SparseArrays
	@reexport using OffsetArrays, FITSIO,Printf,JLD2

	function source(ω,ℓ;r_src=Rsun-75e5)

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

	# struct Gfn
	# 	mode :: NamedTuple{(:ω, :ω_ind, :ℓ),Tuple{Float64,Int64,Int64}}
	# 	G :: OffsetArray{ComplexF64,3,Array{ComplexF64,3}}
	# end

	function compute_radial_component_of_Gfn_onemode(ω,ℓ;r_src=Rsun-75e5,tangential=false)

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

	function compute_Greenfn_components_allmodes_parallel(r_src=Rsun-75e5;ℓ_arr=1:100,
		ν_low=2.0e-3,ν_high=4.5e-3,num_ν=1250)

		Nν = 3*num_ν; Nt = 2*(Nν-1)
		ν_full = LinRange(0,7.5e-3,Nν); ν_nyquist = ν_full[end];
		ν_arr = ν_full[ν_low .≤ ν_full .≤ ν_high];
		dν = ν_full[2] - ν_full[1]; T=1/dν; dt = T/Nt;
		ν_start_zeros = count(ν_full .< ν_low)
		ν_end_zeros = count(ν_high .< ν_full)
		println("$(length(ν_arr)) frequencies over $(@sprintf "%.1e" ν_arr[1]) to $(@sprintf "%.1e" ν_arr[end])")

		ω_arr = 2π .* ν_arr;

		println("ℓ from $(ℓ_arr[1]) to $(ℓ_arr[end])")

		Gfn_save_directory = Gfn_path_from_source_radius(r_src)

		if !isdir(Gfn_save_directory)
			mkdir(Gfn_save_directory)
		end
		println("Saving output to $Gfn_save_directory")

		w = workers_active(ℓ_arr,ν_arr)
		num_procs = length(w)

		println("Number of workers: $num_procs")

		function compute_G_somemodes_serial_oneproc(rank)

			# Each processor will save all ℓ's  for a range of frequencies if number of frequencies can be split evenly across processors.
			ℓ_ω_proc = split_product_across_processors(ℓ_arr,axes(ν_arr,1),nworkers(),rank);
			
			# Gfn_arr = Vector{Gfn}(undef,length(ℓ_ω_proc));
			save_path = joinpath(Gfn_save_directory,@sprintf "Gfn_proc_%03d.fits" rank)

			# file = jldopen(save_path,"w")
			file = FITS(save_path,"w")

			# save real and imaginary parts separately 
			G = OffsetArray(zeros(nr,2,2,2,length(ℓ_ω_proc)),0,0,-1,-1,0)

			Dr = dbydr(dr) # derivative operator
			T2 = (N2./g .+ Dr*log.(r.^2 .*ρ) )

			for (ind,(ℓ,ω_ind)) in enumerate(ℓ_ω_proc)
				
				ω = ω_arr[ω_ind]
				αr,βr,αh,βh = compute_radial_component_of_Gfn_onemode(ω,ℓ);

				@. G[:,1,0,0,ind] = real(αr)
				@. G[:,2,0,0,ind] = imag(αr)
				@. G[:,1,1,0,ind] = real(Ω(ℓ,0) * βr/(ρ*r*ω^2))
				@. G[:,2,1,0,ind] = imag(Ω(ℓ,0) * βr/(ρ*r*ω^2))

				@. G[:,1,0,1,ind] = real(αh) / √2
				@. G[:,2,0,1,ind] = imag(αh) / √2
				
				G[:,1,1,1,ind] .=  real(r./√(ℓ*(ℓ+1)) .* (βh./(ρ.*c.^2) .+ T2.*αh + Dr*αh) ./2)
				G[:,2,1,1,ind] .=  imag(r./√(ℓ*(ℓ+1)) .* (βh./(ρ.*c.^2) .+ T2.*αh + Dr*αh) ./2)
				
			end

			write(file,G.parent)

			# modes_ℓ,modes_ω_ind = collect.(zip(ℓ_ω_proc...))

			# write(file,Dict("mode_l"=>modes_ℓ,"mode_omega_ind"=>modes_ω_ind))

			close(file)
			return rank
		end

		if isempty(w)
			return
		else
			f = [@spawn compute_G_somemodes_serial_oneproc(rank) for rank in 1:length(w)]
			println("$(length(fetch.(f))) jobs finished")
		end

		@save joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs ν_full dν ν_start_zeros ν_end_zeros Nν Nt dt ν_nyquist T
	end

	function compute_Greenfn_radial_components_allmodes_parallel(r_src=Rsun-75e5;ℓ_arr=1:100,
		ν_low=2.0e-3,ν_high=4.5e-3,num_ν=1250)

		Nν = 3*num_ν; Nt = 2*(Nν-1)
		ν_full = LinRange(0,7.5e-3,Nν); ν_nyquist = ν_full[end];
		ν_arr = ν_full[ν_low .≤ ν_full .≤ ν_high];
		dν = ν_full[2] - ν_full[1]; T=1/dν; dt = T/Nt;
		ν_start_zeros = count(ν_full .< ν_low)
		ν_end_zeros = count(ν_high .< ν_full)
		println("$(length(ν_arr)) frequencies over $(@sprintf "%.1e" ν_arr[1]) to $(@sprintf "%.1e" ν_arr[end])")

		ω_arr = 2π .* ν_arr;

		println("ℓ from $(ℓ_arr[1]) to $(ℓ_arr[end])")

		Gfn_save_directory = Gfn_path_from_source_radius(r_src)

		if !isdir(Gfn_save_directory)
			mkdir(Gfn_save_directory)
		end
		println("Saving output to $Gfn_save_directory")

		w = workers_active(ℓ_arr,ν_arr)
		num_procs = length(w)

		println("Number of workers: $num_procs")

		function compute_G_somemodes_serial_oneproc(rank)

			# Each processor will save all ℓ's  for a range of frequencies if number of frequencies can be split evenly across processors.
			ℓ_ω_proc = split_product_across_processors(ℓ_arr,axes(ν_arr,1),nworkers(),rank);
			
			# Gfn_arr = Vector{Gfn}(undef,length(ℓ_ω_proc));
			save_path = joinpath(Gfn_save_directory,@sprintf "Gfn_proc_%03d.fits" rank)

			# file = jldopen(save_path,"w")
			file = FITS(save_path,"w")

			# save real and imaginary parts separately 
			G = OffsetArray(zeros(nr,2,2,1,length(ℓ_ω_proc)),0,0,-1,-1,0)

			Dr = dbydr(dr) # derivative operator
			T2 = (N2./g .+ Dr*log.(r.^2 .*ρ) )

			for (ind,(ℓ,ω_ind)) in enumerate(ℓ_ω_proc)
				
				ω = ω_arr[ω_ind]
				αr,βr, = compute_radial_component_of_Gfn_onemode(ω,ℓ,tangential=false);

				@. G[:,1,0,0,ind] = real(αr)
				@. G[:,2,0,0,ind] = imag(αr)
				@. G[:,1,1,0,ind] = real(Ω(ℓ,0) * βr/(ρ*r*ω^2))
				@. G[:,2,1,0,ind] = imag(Ω(ℓ,0) * βr/(ρ*r*ω^2))
				
			end

			write(file,G.parent)

			# modes_ℓ,modes_ω_ind = collect.(zip(ℓ_ω_proc...))

			# write(file,Dict("mode_l"=>modes_ℓ,"mode_omega_ind"=>modes_ω_ind))

			close(file)
			return rank
		end

		if isempty(w)
			return
		else
			f = [@spawn compute_G_somemodes_serial_oneproc(rank) for rank in 1:length(w)]
			println("$(length(fetch.(f))) jobs finished")
		end

		@save joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs ν_full dν ν_start_zeros ν_end_zeros Nν Nt dt ν_nyquist T
	end

	export compute_Greenfn_radial_components_allmodes_parallel,Gfn,Gfn_path_from_source_radius,Ω
end

################################################################
# Three dimensional Green function and cross-covariances
################################################################

module Greenfn_3D

	using Reexport,Interpolations,FFTW,FastGaussQuadrature,DSP
	@reexport using Main.Greenfn_radial
	import Main.Greenfn_radial: Gfn_path_from_source_radius
	@reexport using Legendre,PointsOnASphere,TwoPointFunctions,VectorFieldsOnASphere

	Gfn_path_from_source_radius(x::Point3D) = Gfn_path_from_source_radius(x.r)

	function Powspec(ω)
		σ = 2π*0.4e-3
		ω0 = 2π*3e-3
		exp(-(ω-ω0)^2/(2σ^2))
	end

	function load_Greenfn_radial_coordinates_onefreq(Gfn_save_directory,ν,ν_arr,ℓ_arr,num_procs)

    	ν_test_index = argmin(abs.(ν_arr .- ν));

		proc_id_start = get_processor_id_from_split_array(ℓ_arr,axes(ν_arr,1),(ℓ_arr[1],ν_test_index),num_procs);
	    proc_id_end = get_processor_id_from_split_array(ℓ_arr,axes(ν_arr,1),(ℓ_arr[end],ν_test_index),num_procs);

	    Gfn_arr_onefreq = Vector{Gfn}(undef,length(ℓ_arr))

	    Gfn_ℓ_index = 1
	    for proc_id in proc_id_start:proc_id_end
	    	G_proc_file = joinpath(Gfn_save_directory,@sprintf "Gfn_proc_%03d.jld2" proc_id)
	    	@load G_proc_file Gfn_arr
	    	for G in Gfn_arr
	    		
	    		if G.mode.ω_ind == ν_test_index
		    		Gfn_arr_onefreq[Gfn_ℓ_index] = G
		    		Gfn_ℓ_index += 1
	    		end

	    		if Gfn_ℓ_index > length(ℓ_arr)
	    			break
	    		end
	    	end
	    end
	    return Gfn_arr_onefreq
    end

    function compute_3D_Greenfn_helicity_angular_sections_onefreq(x′,(nθ,nϕ)::NTuple{2,Int64},ν::Real=3e-3,procid=myid()-1)

		Gfn_save_directory = Gfn_path_from_source_radius(x′)

    	@load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs
    	
    	Gfn_radial_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_save_directory,ν,ν_arr,ℓ_arr,num_procs)

    	ℓmax = ℓ_arr[end]

    	θ_full = LinRange(0,π,nθ) # might need non-uniform grid in θ    	
    	ϕ_full = LinRange(0,2π,nϕ)

    	θ_ϕ_iterator = split_product_across_processors(θ_full,ϕ_full,nworkers(),procid)

    	function compute_3D_Greenfn_helicity_angular_sections_onefreq(x′,Gfn_radial_arr,θ_ϕ_iterator,ℓmax::Integer,ν::Real=3e-3,procid=myid()-1)

	    	Gfn_3D_arr = zeros(ComplexF64,1:nr,1:length(θ_ϕ_iterator),-1:1)

	    	d01Pl_cosχ = OffsetArray{Float64}(undef,0:ℓmax,0:1)
	    	Pl_cosχ = view(d01Pl_cosχ,:,0)
	    	dPl_cosχ = view(d01Pl_cosχ,:,1)

	    	for (θϕ_ind,(θ,ϕ)) in enumerate(θ_ϕ_iterator)
	    		
	    		Pl_dPl!( d01Pl_cosχ, cosχ((θ,ϕ),x′) )
		    	
		    	for Gfn in Gfn_radial_arr

		    		ℓ = Gfn.mode.ℓ 

		    		(ℓ<1) && continue

		    		G00 = view(Gfn.G,:,1)
		    		G10 = view(Gfn.G,:,2)

		    		# (-1,0) component
		    		em1_dot_∇_n_dot_n′ = 1/√2 * (∂θ₁cosχ((θ,ϕ),x′) + im * ∇ϕ₁cosχ((θ,ϕ),x′) )	    		
		    		@. Gfn_3D_arr[1:nr,θϕ_ind,-1] += (2ℓ +1)/4π  * G10/Ω(ℓ,0) * dPl_cosχ[ℓ] * em1_dot_∇_n_dot_n′

		    		# (0,0) component
		    		@. Gfn_3D_arr[1:nr,θϕ_ind,0] +=  (2ℓ +1)/4π * G00 * Pl_cosχ[ℓ]
		    		
		    		# (1,0) component
		    		e1_dot_∇_n_dot_n′ = 1/√2 * (-∂θ₁cosχ((θ,ϕ),x′) + im * ∇ϕ₁cosχ((θ,ϕ),x′) )
		    		@. Gfn_3D_arr[1:nr,θϕ_ind,1] +=  (2ℓ +1)/4π  * G10/Ω(ℓ,0) * dPl_cosχ[ℓ] * e1_dot_∇_n_dot_n′

		    	end
		    end

		    return Gfn_3D_arr
	    end

	    compute_3D_Greenfn_helicity_angular_sections_onefreq(x′,Gfn_radial_arr,θ_ϕ_iterator,ℓmax,ν,procid)
    end

    function compute_3D_Greenfn_spherical_angular_sections_onefreq(x′,
    	θ_full::AbstractArray,ϕ_full::AbstractArray,ν::Real=3e-3,procid=myid()-1)

		Gfn_save_directory = Gfn_path_from_source_radius(x′)

    	@load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs
    	
    	Gfn_radial_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_save_directory,ν,ν_arr,ℓ_arr,num_procs)

    	ℓmax = ℓ_arr[end]

    	θ_ϕ_iterator = split_product_across_processors(θ_full,ϕ_full,nworkers(),procid)

    	function compute_3D_Greenfn_spherical_angular_sections_onefreq(x′,Gfn_radial_arr,θ_ϕ_iterator,ℓmax::Integer,ν::Real=3e-3,procid=myid()-1)

	    	Gfn_3D_arr = zeros(ComplexF64,nr,length(θ_ϕ_iterator),3)

	    	d01Pl_cosχ = OffsetArray{Float64}(undef,0:ℓmax,0:1)
	    	Pl_cosχ = view(d01Pl_cosχ,:,0)
	    	dPl_cosχ = view(d01Pl_cosχ,:,1)

	    	for (θϕ_ind,(θ,ϕ)) in enumerate(θ_ϕ_iterator)
	    		
	    		Pl_dPl!( d01Pl_cosχ, cosχ((θ,ϕ),x′) )
		    	
		    	for Gfn in Gfn_radial_arr
		    		
		    		ℓ = Gfn.mode.ℓ 

		    		(ℓ<1) && continue

		    		G00 = view(Gfn.G,:,1)
		    		G10 = view(Gfn.G,:,2)

		    		# (r,r) component
		    		@. Gfn_3D_arr[1:nr,θϕ_ind,1] +=  (2ℓ+1)/4π * G00 * Pl_cosχ[ℓ]

		    		# (θ,r) component
		    		eθ_dot_∇_n_dot_n′ = ∂θ₁cosχ((θ,ϕ),x′)
		    		@. Gfn_3D_arr[1:nr,θϕ_ind,2] += (2ℓ+1)/4π * G10/Ω(ℓ,0) * dPl_cosχ[ℓ] * eθ_dot_∇_n_dot_n′
		    		
		    		# (ϕ,r) component
		    		eϕ_dot_∇_n_dot_n′ = ∇ϕ₁cosχ((θ,ϕ),x′)
		    		@. Gfn_3D_arr[1:nr,θϕ_ind,3] +=  (2ℓ+1)/4π * G10/Ω(ℓ,0) * dPl_cosχ[ℓ] * eϕ_dot_∇_n_dot_n′

		    	end
		    end

		    return Gfn_3D_arr
	    end

	    compute_3D_Greenfn_spherical_angular_sections_onefreq(x′,Gfn_radial_arr,θ_ϕ_iterator,ℓmax,ν,procid) 
    end

    function uϕ∇ϕG_uniform_rotation_angular_sections_onefreq(x′,
    	θ_full::AbstractArray,ϕ_full::AbstractArray,ν::Real=3e-3,procid=myid()-1)

		Gfn_save_directory = Gfn_path_from_source_radius(x′)

    	@load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs
    	
    	Gfn_radial_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_save_directory,ν,ν_arr,ℓ_arr,num_procs)

    	ℓmax = ℓ_arr[end]

    	θ_ϕ_iterator = split_product_across_processors(θ_full,ϕ_full,nworkers(),procid)

    	uϕ∇ϕG = zeros(ComplexF64,nr,length(θ_ϕ_iterator),3)

    	d02Pl_cosχ = OffsetArray{Float64}(undef,0:ℓmax,0:2)
    	Pl = view(d02Pl_cosχ,:,0)
    	dPl = view(d02Pl_cosχ,:,1)
    	d2Pl = view(d02Pl_cosχ,:,2)

    	for (θϕ_ind,(θ,ϕ)) in enumerate(θ_ϕ_iterator)
    		
    		Legendre.Pl_dPl_d2Pl!( d02Pl_cosχ, cosχ((θ,ϕ),x′) )
	    	
	    	for Gfn in Gfn_radial_arr
	    		
	    		ℓ = Gfn.mode.ℓ 

	    		(ℓ<1) && continue

	    		G00 = view(Gfn.G,:,1)
	    		G10 = view(Gfn.G,:,2)

	    		# (r,r) component
	    		∂ϕPl = dPl[ℓ] * ∂ϕ₁cosχ((θ,ϕ),x′)
	    		@. uϕ∇ϕG[:,θϕ_ind,1] +=  (2ℓ+1)/4π * (G00-G10/Ω(ℓ,0)) * ∂ϕPl
	    		# @. uϕ∇ϕG[:,θϕ_ind,1] +=  (2ℓ+1)/4π * G00 * ∂ϕPl
	    		
	    		# (θ,r) component
	    		d2Pl_∂θcosχ_∂ϕcosχ = d2Pl[ℓ] * ∂ϕ₁cosχ((θ,ϕ),x′) * ∂θ₁cosχ((θ,ϕ),x′)
	    		@. uϕ∇ϕG[:,θϕ_ind,2] += (2ℓ+1)/4π  * G10/Ω(ℓ,0) * d2Pl_∂θcosχ_∂ϕcosχ

	    		# (ϕ,r) component
	    		dϕ∇ϕPl = d2Pl[ℓ] * ∇ϕ₁cosχ((θ,ϕ),x′) * ∂ϕ₁cosχ((θ,ϕ),x′) + dPl[ℓ] * ∇ϕ₁∂ϕ₁cosχ((θ,ϕ),x′)
	    		∂θPl =  dPl[ℓ] * ∂θ₁cosχ((θ,ϕ),x′)
	    		@. uϕ∇ϕG[:,θϕ_ind,3] +=  (2ℓ+1)/4π  * (G10/Ω(ℓ,0) * (dϕ∇ϕPl + cos(θ) * ∂θPl ) + G00 * sin(θ) * Pl[ℓ])
	    		# @. uϕ∇ϕG[:,θϕ_ind,3] +=  (2ℓ+1)/4π  * (G00 * sin(θ) * Pl[ℓ])

	    	end
	    end

	    return uϕ∇ϕG
    end

    function δC_uniform_rotation_helicity_angular_sections_onefreq(x1,x2,nθ,nϕ,ν=3e-3,procid=myid()-1)

    	Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

		@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs
		ℓmax = ℓ_arr[end]

		θ_full = LinRange(0,π,nθ)
		ϕ_full = LinRange(0,2π,nϕ)
		θ_ϕ_iterator = split_product_across_processors(θ_full,ϕ_full,nworkers(),procid)

		ν_ind = argmin(abs.(ν_arr .- ν))
		ω = 2π*ν_arr[ν_ind]

		Gfn_radial_arr_onefreq = load_Greenfn_radial_coordinates_onefreq(Gfn_directory_x1,ν,ν_arr,ℓ_arr,num_procs)

		Gfn3D_x1 = compute_3D_Greenfn_helicity_angular_sections_onefreq(x1,Gfn_radial_arr_onefreq,θ_ϕ_iterator,ℓmax,ν,procid)
		Gfn3D_x2 = compute_3D_Greenfn_helicity_angular_sections_onefreq(x2,Gfn_radial_arr_onefreq,θ_ϕ_iterator,ℓmax,ν,procid)

	    Yℓ1 = OffsetArray{ComplexF64}(undef,0:ℓmax,-2:2)
	    Yℓ2 = OffsetArray{ComplexF64}(undef,0:ℓmax,-2:2)

	    r₁_ind = argmin(abs.(r .- x1.r))
	    r₂_ind = argmin(abs.(r .- x2.r))

		integrand = zeros(ComplexF64,nr,length(θ_ϕ_iterator))

		for (θϕ_ind,(θ,ϕ)) in enumerate(θ_ϕ_iterator)

			ϕ₁,θ₁,_ = add_rotations_euler_angles((-x1.ϕ,-x1.θ,0),(0,θ,ϕ))
			ϕ₂,θ₂,_ = add_rotations_euler_angles((-x2.ϕ,-x2.θ,0),(0,θ,ϕ))

			Ylm!(Yℓ1,θ₁,ϕ₁)
			Ylm!(Yℓ2,θ₂,ϕ₂)

				for Gfn in Gfn_radial_arr_onefreq
					ℓ = Gfn.mode.ℓ

					(ℓ < 2) && continue

					Gℓ = OffsetArray(Gfn.G,(0,-1))
					Gℓ[:,1] ./= Ω(ℓ,1)

					for α=-1:1
						@. integrand[1:nr,θϕ_ind] += sin(θ) * √((2ℓ+1)/4π) * (

						Gfn3D_x2[α,1:nr,θϕ_ind]*conj(Gfn.G[r₁_ind,1])*
						((Gfn.G[1:nr,abs(α)] * Ω(ℓ,α) - (α != -1 ? Gfn.G[1:nr,abs(α-1)] : 0) )*Yℓ1[ℓ,α-1] + 
						(Gfn.G[1:nr,abs(α)] * Ω(ℓ,-α) - (α != 1 ? Gfn.G[1:nr,abs(α+1)] : 0) )*Yℓ1[ℓ,α+1]) +
						
						conj(Gfn3D_x1[α,1:nr,θϕ_ind])*Gfn.G[r₂_ind,1]*
						((conj(Gfn.G[1:nr,abs(α)]) * Ω(ℓ,-α) - (α != 1 ? conj(Gfn.G[1:nr,abs(α+1)]) : 0) )*conj(Yℓ2[ℓ,α+1]) + 
						(conj(Gfn.G[1:nr,abs(α)]) * Ω(ℓ,α) - (α != -1 ? conj(Gfn.G[1:nr,abs(α-1)]) : 0) )*conj(Yℓ2[ℓ,α-1]) )

						)

					end

			end
		end

		return integrate.simps(ρ .* r.^2 .* integrand,x=r)
    end

    function δC_uniform_rotation_spherical_angular_sections_onefreq(x1,x2,θ_full,ϕ_full,ν=3e-3,procid=myid()-1)

    	Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

		@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs
		ℓmax = ℓ_arr[end]

		P1 = OffsetArray{Float64}(undef,0:ℓmax,0:2)
		P2 = OffsetArray{Float64}(undef,0:ℓmax,0:2)

		∂ϕPl1 = OffsetArray{Float64}(undef,0:ℓmax)
		∂ϕPl2 = OffsetArray{Float64}(undef,0:ℓmax)
		∇ϕPl1 = OffsetArray{Float64}(undef,0:ℓmax)
		∇ϕPl2 = OffsetArray{Float64}(undef,0:ℓmax)
		∂θPl1 = OffsetArray{Float64}(undef,0:ℓmax)
		∂θPl2 = OffsetArray{Float64}(undef,0:ℓmax)
		∂ϕ∂θPl1 = OffsetArray{Float64}(undef,0:ℓmax)
		∂ϕ∂θPl2 = OffsetArray{Float64}(undef,0:ℓmax)
		∂²ϕPl1 = OffsetArray{Float64}(undef,0:ℓmax)
		∂²ϕPl2 = OffsetArray{Float64}(undef,0:ℓmax)
		∇ϕ∂ϕPl1 = OffsetArray{Float64}(undef,0:ℓmax)
		∇ϕ∂ϕPl2 = OffsetArray{Float64}(undef,0:ℓmax)

		θ_ϕ_iterator = split_product_across_processors(θ_full,ϕ_full,nworkers(),procid)

		ν_ind = argmin(abs.(ν_arr .- ν))
		ω = 2π*ν_arr[ν_ind]

		Gfn_radial_arr_onefreq = load_Greenfn_radial_coordinates_onefreq(Gfn_directory_x1,ν,ν_arr,ℓ_arr,num_procs)

		Gfn3D_x1 = compute_3D_Greenfn_spherical_angular_sections_onefreq(x1,Gfn_radial_arr_onefreq,θ_ϕ_iterator,ℓmax,ν,procid)
		Gfn3D_x2 = compute_3D_Greenfn_spherical_angular_sections_onefreq(x2,Gfn_radial_arr_onefreq,θ_ϕ_iterator,ℓmax,ν,procid)

	    r₁_ind = argmin(abs.(r .- x1.r))
	    r₂_ind = argmin(abs.(r .- x2.r))

		integrand = zeros(ComplexF64,nr,length(θ_ϕ_iterator))

		for (θϕ_ind,(θ,ϕ)) in enumerate(θ_ϕ_iterator)

			Legendre.Pl_dPl_d2Pl!(P1,cosχ((θ,ϕ),x1),ℓmax=ℓmax)
			Pl1 = view(P1,:,0)
			dPl1 = view(P1,:,1)
			d2Pl1 = view(P1,:,2)

			∂θPl1 .= dPl1 .* ∂θ₁cosχ((θ,ϕ),x1)
			∂ϕPl1 .= dPl1 .* ∂ϕ₁cosχ((θ,ϕ),x1)
			∇ϕPl1 .= dPl1 .* ∇ϕ₁cosχ((θ,ϕ),x1)
			∂ϕ∂θPl1 .= d2Pl1 .* ∂ϕ₁cosχ((θ,ϕ),x1) .* ∂θ₁cosχ((θ,ϕ),x1) .+ dPl1 .* ∂θ₁∂ϕ₁cosχ((θ,ϕ),x1)
			∂²ϕPl1 .= d2Pl1 .* ∂ϕ₁cosχ((θ,ϕ),x1)^2 .+ dPl1 .* ∂²ϕ₁cosχ((θ,ϕ),x1)
			∇ϕ∂ϕPl1 .= d2Pl1 .* ∇ϕ₁cosχ((θ,ϕ),x1) .* ∂ϕ₁cosχ((θ,ϕ),x1) .+ dPl1 .* ∇ϕ₁∂ϕ₁cosχ((θ,ϕ),x1)

			Legendre.Pl_dPl_d2Pl!(P2,cosχ((θ,ϕ),x2),ℓmax=ℓmax)
			Pl2 = view(P2,:,0)
			dPl2 = view(P2,:,1)
			d2Pl2 = view(P2,:,2)

			∂θPl2 .= dPl2 .* ∂θ₁cosχ((θ,ϕ),x2)
			∂ϕPl2 .= dPl2 .* ∂ϕ₁cosχ((θ,ϕ),x2)
			∇ϕPl2 .= dPl2 .* ∇ϕ₁cosχ((θ,ϕ),x2)
			∂ϕ∂θPl2 .= d2Pl2 .* ∂ϕ₁cosχ((θ,ϕ),x2) .* ∂θ₁cosχ((θ,ϕ),x2) .+ dPl2 .* ∂θ₁∂ϕ₁cosχ((θ,ϕ),x2)
			∂²ϕPl2 .= d2Pl2 .* ∂ϕ₁cosχ((θ,ϕ),x2)^2 .+ dPl2 .* ∂²ϕ₁cosχ((θ,ϕ),x2)
			∇ϕ∂ϕPl2 .= d2Pl2 .* ∇ϕ₁cosχ((θ,ϕ),x2) .* ∂ϕ₁cosχ((θ,ϕ),x2) .+ dPl2 .* ∇ϕ₁∂ϕ₁cosχ((θ,ϕ),x2)


			for Gfn in Gfn_radial_arr_onefreq
				ℓ = Gfn.mode.ℓ

				(ℓ < 1) && continue

				G00 = view(Gfn.G,:,1)
				G10 = view(Gfn.G,:,2)

				@. integrand[1:nr,θϕ_ind] += (2ℓ+1)/4π * (

				- G00[r₂_ind]*(
					conj(Gfn3D_x1[:,θϕ_ind,1])*conj(G00 - G10/Ω(ℓ,0)) * ∂ϕPl2[ℓ]  + 
					conj(Gfn3D_x1[:,θϕ_ind,2])*conj(G10/Ω(ℓ,0)) * (∂ϕ∂θPl2[ℓ] - cos(θ)*∇ϕPl2[ℓ]) +
					conj(Gfn3D_x1[:,θϕ_ind,3])*(conj(G10/Ω(ℓ,0))* (∇ϕ∂ϕPl2[ℓ] + cos(θ) * ∂θPl2[ℓ]) + conj(G00)*sin(θ)*Pl2[ℓ])
					) +

				conj(G00[r₁_ind])*(
					Gfn3D_x2[:,θϕ_ind,1]*(G00 - G10/Ω(ℓ,0)) * ∂ϕPl1[ℓ]  + 
					Gfn3D_x2[:,θϕ_ind,2]*G10/Ω(ℓ,0) * (∂ϕ∂θPl1[ℓ] - cos(θ)*∇ϕPl1[ℓ]) +
					Gfn3D_x2[:,θϕ_ind,3]*(G10/Ω(ℓ,0)* (∇ϕ∂ϕPl1[ℓ] + cos(θ) * ∂θPl1[ℓ]) + G00*sin(θ)*Pl1[ℓ] ) 
					)
				)
			
			end

		end

		return integrate.simps(ρ .* r.^2 .* integrand,x=r)
    end

	function compute_3D_Greenfn_components(x′::Point3D,θ::AbstractArray,ϕ::AbstractArray,ν::Real=3e-3;basis="spherical")
		
		nθ = length(θ); 
		nϕ=length(ϕ);
		
		f = eval(Symbol("compute_3D_Greenfn_$(basis)_angular_sections_onefreq"))

		Gfn_3D_futures_procs = [@spawnat p f(x′,θ,ϕ,ν) for p in workers_active(1:nθ,1:nϕ)]

		Gfn_3D = reshape(cat(fetch.(Gfn_3D_futures_procs)...,dims=2),nr,nθ,nϕ,3)
	end

	function compute_u_dot_∇_G_uniform_rotation(x′::Point3D,θ::AbstractArray,ϕ::AbstractArray,ν::Real=3e-3)
		Gfn_save_directory = Gfn_path_from_source_radius(x′)

		# @load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs

		# ℓmax = ℓ_arr[end]
		
		nθ = length(θ); 
		nϕ=length(ϕ);
		
		Gfn_3D_futures_procs = [@spawnat p uϕ∇ϕG_uniform_rotation_angular_sections_onefreq(x′,θ,ϕ,ν) for p in workers_active(1:nθ,1:nϕ)]

		Gfn_3D = reshape(cat(fetch.(Gfn_3D_futures_procs)...,dims=2),nr,nθ,nϕ,3)
	end

	function δLG_uniform_rotation(x_src::Point3D,θ::AbstractArray,ϕ::AbstractArray,ν::Real=3e-3)
		
		Gfn_save_directory = Gfn_path_from_source_radius(x_src)
		@load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs

		ν = ν_arr[argmin(abs.(ν_arr .- ν))]; ω = 2π*ν

		udot∇G = compute_u_dot_∇_G_uniform_rotation(x_src,θ,ϕ,ν)

		return @. -2im*ω*ρ*udot∇G
	end

	function δGrr_uniform_rotation_firstborn(x1 = Point3D(Rsun-75e5,π/2,0),x_src=Point3D(Rsun-75e5,π/2,π/3),
		ν::Real=3e-3)
		# δG_ik(x1,x2) = -∫dx Gij(x1,x)[δL(x)G(x,x2)]jk = -∫dx Gji(x,x1)[δL(x)G(x,x2)]jk
		# We compute δG_rr(x1,x2) = -∫dx Gjr(x,x1)[δL(x)G(x,x2)]jr

		Gfn_path = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs

		ν_on_grid = ν_arr[argmin(abs.(ν_arr .- ν))]

		@printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_arr[1] ℓ_arr[end]
		ℓmax = ℓ_arr[end]
		
		# θ = LinRange(0,π,2ℓmax)[2:end-1]; 
		cosθGL,wGL = gausslegendre(ℓmax); θ=acos.(cosθGL);
		nθ = length(θ); dθ = θ[2] - θ[1];
		ϕ = LinRange(0,2π,2ℓmax); dϕ = ϕ[2] - ϕ[1]; nϕ=length(ϕ)
		θ3D = reshape(θ,1,nθ,1)

		δLG = δLG_uniform_rotation(x_src,θ,ϕ,ν_on_grid)
		G = compute_3D_Greenfn_components(x1,θ,ϕ,ν_on_grid,basis="spherical")
		integrand = dropdims(sum(G .* δLG, dims=4),dims=4)

		dG = - integrate.simps(dropdims(sum(wGL .* integrate.simps((@. r^2*integrand),x=r,axis=0),dims=1),dims=1),dx=dϕ)
	end

	function δGrr_uniform_rotation_firstborn_integrated_over_angle(x1= Point3D(Rsun-75e5,π/2,0),x_src=Point3D(Rsun-75e5,π/2,π/3),
		ν::Real=3e-3)
		# δG_ik(x1,x2) = -∫dx Gij(x1,x)[δL(x)G(x,x2)]jk = -∫dx Gji(x,x1)[δL(x)G(x,x2)]jk
		# We compute δG_rr(x1,x2) = -∫dx Gjr(x,x1)[δL(x)G(x,x2)]jr

		Gfn_path = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs

		ν_test_index = argmin(abs.(ν_arr .- ν))
		ν_on_grid = ν_arr[ν_test_index]
		ω = 2π * ν_on_grid

		@printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_arr[1] ℓ_arr[end]
		ℓmax = ℓ_arr[end]

		Gfn_radial_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_path,ν,ν_arr,ℓ_arr,num_procs)

		dPl_cosχ = dPl(cosχ(x1,x_src),ℓmax=ℓ_arr[end])

		Gfn_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_path,ν,ν_arr,ℓ_arr,num_procs)

		δG = zeros(ComplexF64,nr)
	    	
		for Gfn in Gfn_arr
			ℓ = Gfn.mode.ℓ
			ω_ind = Gfn.mode.ω_ind

		    G0 = view(Gfn.G,:,1)
		    G1 = view(Gfn.G,:,2)

		    ((ℓ<1) || (ω_ind != ν_test_index )) && continue

		    δG .+= (2ℓ+1)/4π .* (@. G0^2 - 2G0*G1/Ω(ℓ,0) + (ℓ*(ℓ+1)-1)*(G1/Ω(ℓ,0))^2) .* (dPl_cosχ[ℓ]*∂ϕ₁cosχ(x1,x_src))
		end

		return 2im*ω*integrate.simps((@. r^2 * ρ * δG),x=r)
	end

	function δGrr_uniform_rotation_rotatedwaves(x1::Point3D,x_src::Point3D,ν=3e-3)
		
		# Assuming Ω_rotation = 1, scale the result up by Ω as needed
		Gfn_path = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs

		ν_test_ind = argmin(abs.(ν_arr .- ν))

		ν_on_grid = ν_arr[ν_test_ind]

		@printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_arr[1] ℓ_arr[end]

		ν_ind_range = ν_test_ind-2:ν_test_ind+2
		ω_arr = 2π .* ν_arr[ν_ind_range]
		dω = ω_arr[2] - ω_arr[1]
		
		ℓmax = ℓ_arr[end]

		Grr_rsrc_equator = OffsetArray(zeros(ComplexF64,length(ν_ind_range)),ν_ind_range) # Green function at equator on the surface

		r₁_ind = argmin(abs.(r .- x1.r)) # observations at the source radius, can be anything as long as consistent

		∂ϕ₁Pl_cosχ = dPl(cosχ(x1,x_src),ℓmax=ℓmax) .* ∂ϕ₁cosχ(x1,x_src)

		modes_on_proc = split_product_across_processors(ℓ_arr,ν_ind_range)

		proc_id_range = get_processor_range_from_split_array(ℓ_arr,axes(ν_arr,1),modes_on_proc,num_procs);
		

		function sum_modes(proc_id,ν_ind_range,∂ϕ₁Pl_cosχ)

			G = OffsetArray(zeros(ComplexF64,length(ν_ind_range)),ν_ind_range)

			G_proc_file = joinpath(Gfn_path,@sprintf "Gfn_proc_%03d.jld2" proc_id)
	    	@load G_proc_file Gfn_arr
	    	for Gfn in Gfn_arr

	    		ℓ = Gfn.mode.ℓ
	    		ω_ind = Gfn.mode.ω_ind
			    
			    if (ω_ind ∉ ν_ind_range) || (ℓ<1)
			    	continue
			    end
	    	
	    		G0 = view(Gfn.G,:,1)
	    		
	    		G[ω_ind] += (2ℓ+1)/4π * G0[r₁_ind] * ∂ϕ₁Pl_cosχ[ℓ]
	    		
	    	end

			return G
		end

		Grr_rsrc_equator = @distributed (+) for proc_id in proc_id_start:proc_id_end
								sum_modes(proc_id,ν_ind_range,∂ϕ₁Pl_cosχ)
							end

	    D_op = dbydr(ω_arr,Dict(2=>3)) 
	    # display(Matrix(D_op))

	    ∂ωG = D_op*Grr_rsrc_equator.parent
		return -im*∂ωG[div(length(ν_ind_range),2) + 1]
	end

	function uϕ_dot_∇G_finite_difference(x::Point3D,θ::AbstractArray,ϕ::AbstractArray,ν::Real=3e-3)
		
		dϕ = 1e-8; dϕ_grid = ϕ[2] - ϕ[1]
		θ3D = reshape(θ,1,length(θ),1)
		G = compute_3D_Greenfn_components(x,θ,ϕ,ν,basis="spherical")
		
		G_plus_r = compute_3D_Greenfn_components(Point3D(x.r,x.θ,x.ϕ-dϕ),θ,ϕ,ν,basis="spherical")[:,:,:,1] # rotate source in opposite direction
		G_minus_r = compute_3D_Greenfn_components(Point3D(x.r,x.θ,x.ϕ+dϕ),θ,ϕ,ν,basis="spherical")[:,:,:,1] # rotate source in opposite direction

		G_r = view(G,:,:,:,1)
		G_θ = view(G,:,:,:,2)
		G_ϕ = view(G,:,:,:,3)

		uϕ∇ϕG_r = @. (G_plus_r - G_minus_r)/2dϕ- G_ϕ * sin(θ3D)
		uϕ∇ϕG_θ = (-roll(G_θ,-2) .+ 8roll(G_θ,-1) .- 8roll(G_θ,1) .+ roll(G_θ,2))./12dϕ_grid .- G_ϕ .* cos.(θ3D)
		uϕ∇ϕG_ϕ = (-roll(G_ϕ,-2) .+ 8roll(G_ϕ,-1) .- 8roll(G_ϕ,1) .+ roll(G_ϕ,2))./12dϕ_grid .+ G_θ .* cos.(θ3D) .+ G_r .* sin.(θ3D)
		
		return cat(uϕ∇ϕG_r,uϕ∇ϕG_θ,uϕ∇ϕG_ϕ,dims=4)
	end

	function δLG_uniform_rotation_finite_difference(x::Point3D,θ::AbstractArray,ϕ::AbstractArray,ν::Real=3e-3)
		# This is simply -2iωρ u⋅∇ G
		ω = 2π*ν
		return -2im .*ω.*ρ.*uϕ_dot_∇G_finite_difference(x,θ,ϕ,ν)
	end

	function δGrr_uniform_rotation_firstborn_finite_difference(x1 = Point3D(Rsun-75e5,π/2,0),x2=Point3D(Rsun-75e5,π/2,π/3),ν=3e-3)
		# δG_ik(x1,x2) = -∫dx Gij(x1,x)[δL(x)G(x,x2)]jk = -∫dx Gji(x,x1)[δL(x)G(x,x2)]jk
		# We compute δG_rr(x1,x2) = -∫dx Gjr(x,x1)[δL(x)G(x,x2)]jr

		Gfn_path = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs
		ν = ν_arr[argmin(abs.(ν_arr .- ν))]
		ℓmax = ℓ_arr[end]

		θ = LinRange(0,π,2ℓmax)[2:end-1]; 
		# cosθGL,wGL = gausslegendre(2ℓmax); θ=acos.(cosθGL);
		nθ = length(θ); dθ = θ[2] - θ[1];
		ϕ = LinRange(0,2π,8ℓmax); dϕ = ϕ[2] - ϕ[1];

		G = compute_3D_Greenfn_components(x1,θ,ϕ,ν,basis="spherical");
		δLG = δLG_uniform_rotation_finite_difference(x2,θ,ϕ,ν);
		
		integrand = dropdims(sum(G .* δLG, dims=4),dims=4)
	
		dG = - integrate.simps(integrate.simps(sin.(θ).*integrate.simps((@. r^2*integrand),x=r,axis=0),dx=dθ,axis=0),dx=dϕ)
	end

	function δCω_uniform_rotation_firstborn(x1 = Point3D(Rsun-75e5,π/2,0),x2=Point3D(Rsun-75e5,π/2,π/3),ν=3e-3;Ω_rot = 20e2/Rsun)
		
		Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

		@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs

		ν_on_grid = ν_arr[argmin(abs.(ν_arr .- ν))]
		ω = 2π*ν_on_grid

		ℓmax = ℓ_arr[end]
		nθ,nϕ = 2ℓmax,2ℓmax

		θ = LinRange(0,π,nθ)
		ϕ = LinRange(0,2π,nϕ)

		workers_active = [i for i in workers() if length(split_product_across_processors(1:nθ,1:nϕ,nworkers(),i-1))!=0]

		dC_Ω_futures = [@spawnat p δC_uniform_rotation_helicity_angular_sections_onefreq(x1,x2,nθ,nϕ,ν) for p in workers_active]

		dC_Ω = reshape(vcat(fetch.(dC_Ω_futures)...),nθ,nϕ)

		dC = -√2 *Ω_rot* ω^3 * Powspec(ω) * ∮dΩ(dC_Ω,θ,ϕ)
	end

	function δCω_uniform_rotation_firstborn_krishnendu(x1 = Point3D(Rsun-75e5,π/2,0),x2=Point3D(Rsun-75e5,π/2,π/3),ν=3e-3;Ω_rot = 20e2/Rsun)
		Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

		@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs

		ν_on_grid = ν_arr[argmin(abs.(ν_arr .- ν))]
		ω = 2π*ν_on_grid

		@printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_arr[1] ℓ_arr[end]

		ℓmax = ℓ_arr[end]

		θ = LinRange(0,π,4ℓmax)[2:end-1]; nθ = length(θ)
		cosθGL,wGL = gausslegendre(4ℓmax); θGL=acos.(cosθGL); nθGL = length(θGL);
		ϕ = LinRange(0,2π,4ℓmax); dϕ = ϕ[2] - ϕ[1]; nϕ=length(ϕ)

		# linear θ grid
		workers_active = [i for i in workers() if length(split_product_across_processors(1:nθ,1:nϕ,nworkers(),i-1))!=0]
		dC_Ω_futures = [@spawnat p δC_uniform_rotation_spherical_angular_sections_onefreq(x1,x2,θ,ϕ,ν) for p in workers_active]
		dC_Ω = reshape(vcat(fetch.(dC_Ω_futures)...),nθ,nϕ)

		dC = 2im *Ω_rot* ω^3 * Powspec(ω) * integrate.simps(integrate.simps(dC_Ω.*sin.(θ),x=θ,axis=0),dx=dϕ)
		println("Linear θ grid: δC = $dC")

		# gauss-legendre nodes
		workers_active = [i for i in workers() if length(split_product_across_processors(1:nθGL,1:nϕ,nworkers(),i-1))!=0]
		dC_Ω_futures = [@spawnat p δC_uniform_rotation_spherical_angular_sections_onefreq(x1,x2,θGL,ϕ,ν) for p in workers_active]
		dC_Ω_GL = reshape(vcat(fetch.(dC_Ω_futures)...),nθGL,nϕ)

		dC = 2im *Ω_rot* ω^3 * Powspec(ω) * integrate.simps(dropdims(sum(wGL.*dC_Ω_GL,dims=1),dims=1),dx=dϕ)

		println("Gauss-Legendre quadrature: δC = $dC")
	end

	function δCω_uniform_rotation_firstborn(x1 = Point3D(Rsun-75e5,π/2,0),x2=Point3D(Rsun-75e5,π/2,π/3),ν=3e-3;Ω_rot = 20e2/Rsun)
		
		Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

		@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs

		ν_on_grid = ν_arr[argmin(abs.(ν_arr .- ν))]
		ω = 2π*ν_on_grid

		ℓmax = ℓ_arr[end]
		nθ,nϕ = 2ℓmax,2ℓmax

		θ = LinRange(0,π,nθ)
		ϕ = LinRange(0,2π,nϕ)

		workers_active = [i for i in workers() if length(split_product_across_processors(1:nθ,1:nϕ,nworkers(),i-1))!=0]

		dC_Ω_futures = [@spawnat p δC_uniform_rotation_helicity_angular_sections_onefreq(x1,x2,nθ,nϕ,ν) for p in workers_active]

		dC_Ω = reshape(vcat(fetch.(dC_Ω_futures)...),nθ,nϕ)

		dC = -√2 *Ω_rot* ω^3 * Powspec(ω) * ∮dΩ(dC_Ω,θ,ϕ)
	end

	function δCω_uniform_rotation_firstborn_krishnendu(x1 = Point3D(Rsun-75e5,π/2,0),x2=Point3D(Rsun-75e5,π/2,π/3),ν=3e-3;Ω_rot = 20e2/Rsun)
		Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

		@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs

		ν_on_grid = ν_arr[argmin(abs.(ν_arr .- ν))]
		ω = 2π*ν_on_grid

		@printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_arr[1] ℓ_arr[end]

		ℓmax = ℓ_arr[end]

		θ = LinRange(0,π,4ℓmax)[2:end-1]; nθ = length(θ)
		cosθGL,wGL = gausslegendre(4ℓmax); θGL=acos.(cosθGL); nθGL = length(θGL);
		ϕ = LinRange(0,2π,4ℓmax); dϕ = ϕ[2] - ϕ[1]; nϕ=length(ϕ)

		# linear θ grid
		workers_active = [i for i in workers() if length(split_product_across_processors(1:nθ,1:nϕ,nworkers(),i-1))!=0]
		dC_Ω_futures = [@spawnat p δC_uniform_rotation_spherical_angular_sections_onefreq(x1,x2,θ,ϕ,ν) for p in workers_active]
		dC_Ω = reshape(vcat(fetch.(dC_Ω_futures)...),nθ,nϕ)

		dC = 2im *Ω_rot* ω^3 * Powspec(ω) * integrate.simps(integrate.simps(dC_Ω.*sin.(θ),x=θ,axis=0),dx=dϕ)
		println("Linear θ grid: δC = $dC")

		# gauss-legendre nodes
		workers_active = [i for i in workers() if length(split_product_across_processors(1:nθGL,1:nϕ,nworkers(),i-1))!=0]
		dC_Ω_futures = [@spawnat p δC_uniform_rotation_spherical_angular_sections_onefreq(x1,x2,θGL,ϕ,ν) for p in workers_active]
		dC_Ω_GL = reshape(vcat(fetch.(dC_Ω_futures)...),nθGL,nϕ)

		dC = 2im *Ω_rot* ω^3 * Powspec(ω) * integrate.simps(dropdims(sum(wGL.*dC_Ω_GL,dims=1),dims=1),dx=dϕ)

		println("Gauss-Legendre quadrature: δC = $dC")
	end

end

#################################################################
# Cross covariances and changes in cross covariances
#################################################################

module crosscov
	
	using Reexport,Interpolations,FFTW,FastGaussQuadrature,DSP
	@reexport using Main.Greenfn_radial
	import Main.Greenfn_radial: Gfn_path_from_source_radius
	
	@reexport using PyCall
	@pyimport scipy.integrate as integrate
	import PyPlot; plt=PyPlot

	@reexport using Legendre,PointsOnASphere,TwoPointFunctions,VectorFieldsOnASphere

	export Cω,Cϕω,Cω_onefreq,h,Powspec,Ct
	export δCω_uniform_rotation_firstborn_integrated_over_angle
	export δCω_uniform_rotation_rotatedwaves_linearapprox
	export δCω_uniform_rotation_rotatedwaves,δCt_uniform_rotation_rotatedwaves,δCt_uniform_rotation_rotatedwaves_linearapprox

	Gfn_path_from_source_radius(x::Point3D) = Gfn_path_from_source_radius(x.r)

	function line_of_sight(p::Point3D,detector::Point3D=Point3D(149.6e11,π/2,0))
		point_vector = CartesianVector(p)
		Detector_vector = CartesianVector(detector)
		l = HelicityVector(Detector_vector-point_vector)
		unitvector(l)
	end

	line_of_sight(n::Point2D,detector::Point3D=Point3D(149.6e11,π/2,0)) = line_of_sight(Point3D(Rsun,n),detector)

	function Powspec(ω)
		σ = 2π*0.4e-3
		ω0 = 2π*3e-3
		exp(-(ω-ω0)^2/(2σ^2))
	end

	#######################################################################################################

	# Check if all modes are being covered in parallel

	function parallel_check(;r_obs=Rsun-75e5,ℓ_range=nothing,ν_ind_range=nothing)
		Gfn_path = Gfn_path_from_source_radius(r_obs)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs

		Nν_Gfn = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = intersect(ℓ_arr,ℓ_range)
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(1:Nν_Gfn,ν_ind_range)
		end

		modes_checked = OffsetArray{Bool}(undef,ℓ_range,ν_ind_range)
		fill!(modes_checked,false)

		function singleprocessor(rank)

			modes_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,nworkers(),rank)

			proc_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,modes_on_proc,num_procs)

			modes_checked_proc = OffsetArray{Bool}(undef,ℓ_range,ν_ind_range)
			fill!(modes_checked_proc,false)

			for proc_id in proc_range

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

		    	for (ℓ,ω_ind) in intersect(modes_in_file,modes_on_proc)

		    		# mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))
		    		modes_checked_proc[ℓ,ω_ind] = true
		    		
				end
			end

			return modes_checked_proc
		end

		futures = [@spawnat p singleprocessor(rank) for (rank,p) in enumerate(workers_active(ℓ_range,ν_ind_range))]

		for f in futures
			modes_checked_proc = fetch(f)
			modes_checked .|= modes_checked_proc
		end

		return all(modes_checked)
	end

	#######################################################################################################

	function Cω(x1::Point3D,x2::Point3D;r_src=Rsun-75e5,ℓ_range=nothing,ν_ind_range=nothing)
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs Nν ν_start_zeros

		Cω_arr = zeros(ComplexF64,Nν)
		Nν_Gfn = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = intersect(ℓ_arr,ℓ_range)
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(ν_ind_range,1:Nν_Gfn)
		end

		function Cω_summodes(rank)

			modes_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank) |> collect

			proc_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,modes_on_proc,num_procs)

			ν_ind_min = first(modes_on_proc)[2]
			ν_ind_max = last(modes_on_proc)[2]
			Nν_on_proc = ν_ind_max - ν_ind_min + 1

			Cω_proc = OffsetArray(zeros(ComplexF64,Nν_on_proc),ν_ind_min:ν_ind_max)

			Pl_cosχ = Pl(cosχ(x1,x2),ℓmax=ℓ_arr[end])

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind = argmin(abs.(r .- x2.r))

			for proc_id in proc_range

				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

		    	for (ℓ,ω_ind) in intersect(modes_in_file,modes_on_proc)

		    		mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

		    		@assert(!isnothing(mode_index),"mode index $mode_index for ($ℓ,$ω_ind) not found by splitting across processors")

		    		ω = ν_arr[ω_ind]*2π

		    		G = read(Gsrc_file[1],r₁_ind,:,1,1,mode_index)
		    		α_r₁ = G[1] + im*G[2]


					if r₁_ind == r₂_ind
		    			α_r₂ = α_r₁
		    		else
						G = read(Gsrc_file[1],r₂_ind,:,1,1,mode_index)
			    		α_r₂ = G[1] + im*G[2]
		    		end

		    		Cω_proc[ω_ind] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α_r₁) * α_r₂ * Pl_cosχ[ℓ]
				    
				end

				close(Gsrc_file)
				
			end

			return Cω_proc
		end

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		futures = [@spawnat p Cω_summodes(rank) for (rank,p) in enumerate(procs)]

		for f in futures
			Ci = fetch(f)
			ax = axes(Ci,1)
			@. Cω_arr[ax + ν_start_zeros] += Ci.parent
		end

		return Cω_arr
	end

	function Cω(x1::Point3D,x2_arr::Vector{Point3D};r_src=Rsun-75e5,ℓ_range=nothing,ν_ind_range=nothing)
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs Nν ν_start_zeros

		Cω_arr = zeros(ComplexF64,length(x2_arr),Nν)
		Nν_Gfn = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = intersect(ℓ_arr,ℓ_range)
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(ν_ind_range,1:Nν_Gfn)
		end

		function Cω_summodes(rank)

			modes_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank) |> collect

			proc_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,modes_on_proc,num_procs)

			ν_ind_min = first(modes_on_proc)[2]
			ν_ind_max = last(modes_on_proc)[2]
			Nν_on_proc = ν_ind_max - ν_ind_min + 1

			Cω_proc = zeros(ComplexF64,ν_ind_min:ν_ind_max)

			Pl_cosχ = OffsetArray{Float64}(undef,ℓ_arr[end],1:length(x2_arr))
			for (ind,x2) in enumerate(x2_arr)
				Pl_cosχ[:,ind] = Pl(cosχ(x1,x2),ℓmax=ℓ_arr[end])
			end
			Pl_cosχ = copy(transpose(Pl_cosχ))

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind_prev = argmin(abs.(r .- x2_arr[1].r))

			for proc_id in proc_range

				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

		    	for (ℓ,ω_ind) in intersect(modes_in_file,modes_on_proc)

		    		mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

		    		@assert(!isnothing(mode_index),"mode index $mode_index for ($ℓ,$ω_ind) not found by splitting across processors")

		    		ω = ν_arr[ω_ind]*2π

		    		G = read(Gsrc_file[1],r₁_ind,:,1,1,mode_index)
		    		α_r₁ = G[1] + im*G[2]

		    		for (x2ind,x2) in enumerate(x2_arr)
		    			r₂_ind = argmin(abs.(r .- x2.r))
						if r₁_ind == r₂_ind
			    			α_r₂ = α_r₁
			    		elseif x2ind==1 || r₂_ind != r₂_ind_prev
							G = read(Gsrc_file[1],r₂_ind,:,1,1,mode_index)
				    		α_r₂ = G[1] + im*G[2]
			    		end
			    		r₂_ind_prev = r₂_ind

			    		Cω_proc[x2ind,ω_ind] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α_r₁) * α_r₂ * Pl_cosχ[ℓ]
			    	end
				    
				end

				close(Gsrc_file)
				
			end

			return Cω_proc
		end

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		futures = [@spawnat p Cω_summodes(rank) for (rank,p) in enumerate(procs)]

		for f in futures
			Ci = fetch(f)
			ax = axes(Ci,2)
			@. Cω_arr[:,ax + ν_start_zeros] += Ci.parent
		end

		return copy(transpose(Cω_arr))
	end

	function Cω(n1::Point2D,n2_arr::Vector{Point2D};r_src=Rsun-75e5,r_obs=Rsun-75e5,ℓ_range=nothing,ν_ind_range=nothing)
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs Nν ν_start_zeros

		Cω_arr = zeros(ComplexF64,length(n2_arr),Nν)
		Nν_Gfn = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = intersect(ℓ_arr,ℓ_range)
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(ν_ind_range,1:Nν_Gfn)
		end

		function Cω_summodes(rank)

			modes_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank) |> collect

			proc_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,modes_on_proc,num_procs)

			ν_ind_min = first(modes_on_proc)[2]
			ν_ind_max = last(modes_on_proc)[2]
			Nν_on_proc = ν_ind_max - ν_ind_min + 1

			Cω_proc = zeros(ComplexF64,1:length(n2_arr),ν_ind_min:ν_ind_max)

			Pl_cosχ = OffsetArray{Float64}(undef,0:ℓ_arr[end],1:length(n2_arr))
			for (n2ind,n2) in enumerate(n2_arr)
				Pl_cosχ[:,n2ind] .= Pl(cosχ(n1,n2),ℓmax=ℓ_arr[end])
			end

			Pl_cosχ = copy(transpose(Pl_cosχ))

			robs_ind = argmin(abs.(r .- r_obs))

			for proc_id in proc_range

				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

		    	for (ℓ,ω_ind) in intersect(modes_in_file,modes_on_proc)

		    		mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

		    		@assert(!isnothing(mode_index),"mode index $mode_index for ($ℓ,$ω_ind) not found by splitting across processors")

		    		ω = ν_arr[ω_ind]*2π

		    		G = read(Gsrc_file[1],robs_ind,:,1,1,mode_index)
		    		abs_α_robs² = G[1]^2 + G[2]^2

		    		for n2ind in 1:length(n2_arr)
		    			Cω_proc[n2ind,ω_ind] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * abs_α_robs² * Pl_cosχ[n2ind,ℓ]
		    		end
				    
				end

				close(Gsrc_file)
				
			end

			return Cω_proc
		end

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		futures = [@spawnat p Cω_summodes(rank) for (rank,p) in enumerate(procs)]

		for f in futures
			Ci = fetch(f)
			ax = axes(Ci,2)
			@. Cω_arr[:,ax + ν_start_zeros] += Ci.parent
		end

		return copy(transpose(Cω_arr))
	end

	########################################################################################################
	# Line-of-sight projected cross-covariance
	########################################################################################################

	function Cω_los(x1::Point3D,x2::Point3D;r_src=Rsun-75e5,ℓ_range=nothing,ν_ind_range=nothing)
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs Nν ν_start_zeros

		Cω_arr = zeros(ComplexF64,Nν)
		Nν_Gfn = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = intersect(ℓ_arr,ℓ_range)
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(ν_ind_range,1:Nν_Gfn)
		end

		function Cω_summodes(rank)

			modes_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank) |> collect

			proc_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,modes_on_proc,num_procs)

			ν_ind_min = first(modes_on_proc)[2]
			ν_ind_max = last(modes_on_proc)[2]
			Nν_on_proc = ν_ind_max - ν_ind_min + 1

			Cω_proc = OffsetArray(zeros(ComplexF64,Nν_on_proc),ν_ind_min:ν_ind_max)

			Pl_cosχ = Pl(cosχ(x1,x2),ℓmax=ℓ_arr[end])

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind = argmin(abs.(r .- x2.r))

			# covariant components
			l1 = line_of_sight(x1).components
			l2 = line_of_sight(x2).components

			for proc_id in proc_range

				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)
				# ℓ_in_file = read(file1[2],"mode_l")
				# ω_ind_in_file = read(file1[2],"mode_omega_ind")

				# @assert(all(zip(ℓ_in_file,ω_ind_in_file) .== modes_in_file),"Modes obtained by splitting do not match those in file")

		    	for (ℓ,ω_ind) in intersect(modes_in_file,modes_on_proc)

		    		mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

		    		@assert(!isnothing(mode_index),"mode index $mode_index for ($ℓ,$ω_ind) not found by splitting across processors")
		    		# @assert((ℓ_in_file[mode_index]==ℓ) && (ω_ind_in_file[mode_index]==ω_ind),
		    		# 	"Mode found by splitting does not match that in file. Mode index $mode_index in "*
		    		# 	"file $(@sprintf "Gfn_proc_%03d.fits" proc_id)")
		    		
		    		# @assert(1<=mode_index<=size(file1[1],5),"mode index beyond file limits")

		    		ω = ν_arr[ω_ind]*2π

		    		G = read(Gsrc_file[1],r₁_ind,:,:,1,mode_index)
		    		G0_r₁_rsrc = G[1,1] + im*G[2,1]
		    		G0_r₂_rsrc = G[1,2] + im*G[2,2]

					if r₁_ind == r₂_ind
		    			Cω_proc[ω_ind] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * abs2(G0_r₁_rsrc) * Pl_cosχ[ℓ]
		    		else
						G = read(Gsrc_file[1],r₂_ind,:,:,1,mode_index)
			    		G0_r₂_rsrc = G[1,1] + im*G[2,1]
			    		G1_r₂_rsrc = G[1,2] + im*G[2,2]
			    		Cω_proc[ω_ind] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(G0_r₁_rsrc) * G0_r₂_rsrc * Pl_cosχ[ℓ]
		    		end
				    
				end

				close(Gsrc_file)
				
			end

			return Cω_proc
		end

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		futures = [@spawnat p Cω_summodes(rank) for (rank,p) in enumerate(procs)]

		for f in futures
			Ci = fetch(f)
			ax = axes(Ci,1)
			@. Cω_arr[ax + ν_start_zeros] += Ci.parent
		end

		return Cω_arr
	end

	########################################################################################################
	# Add methods
	########################################################################################################

	for fn in (:Cω,:Cω_los)
		@eval $fn(n1::Point2D,n2::Point2D;r_obs::Real=Rsun-75e5,kwargs...) = $fn(Point3D(r_obs,n1),Point3D(r_obs,n2);kwargs...)
		@eval $fn(Δϕ::Real;r_obs::Real=Rsun-75e5,kwargs...) = $fn(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ);kwargs...)
		@eval $fn(n1::Point2D,n2::Point2D,ν::Real;r_obs::Real=Rsun-75e5,kwargs...) = $fn(Point3D(r_obs,n1),Point3D(r_obs,n2),ν;kwargs...)
		@eval $fn(Δϕ::Real,ν::Real;r_obs::Real=Rsun-75e5,kwargs...) = $fn(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ),ν;kwargs...)
		
		C_single_freq = quote
			function $fn(x1::Point3D,x2::Point3D,ν::Real;r_src=Rsun-75e5,kwargs...)
			
				Gfn_path_src = Gfn_path_from_source_radius(r_src)
				@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ν_start_zeros

				ν_test_ind = argmin(abs.(ν_arr .- ν))

				$fn(x1,x2,ν_ind_range=ν_test_ind:ν_test_ind;kwargs...)[ν_start_zeros + ν_test_ind]
			end
		end
		eval(C_single_freq)
	end

	########################################################################################################
	# Spectrum of C(ℓ,ω)
	########################################################################################################	

	function Cωℓ_spectrum(;ℓ_range=nothing,ν_ind_range=nothing,r_src=Rsun-75e5,r_obs=Rsun-75e5)
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs

		if !isnothing(ℓ_range)
			ℓ_range = intersect(ℓ_arr,ℓ_range)
		else
			ℓ_range = ℓ_arr
		end

		Nν_Gfn = length(ν_arr)

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(ν_ind_range,1:Nν_Gfn)
		end

		robs_ind = argmin(abs.(r .- r_obs))

		function summodes(rank)
			
			modes_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank)
			proc_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,modes_on_proc,num_procs)

			num_modes_proc = length(modes_on_proc)
			all_modes = Iterators.product(ℓ_range,ν_ind_range)
			start_index = get_index_in_split_array(all_modes,first(modes_on_proc))

			Cℓω = OffsetArray{ComplexF64}(undef,start_index:start_index+num_modes_proc-1)

			for proc_id in proc_range
				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(modes_on_proc,modes_in_file)
					
					ℓω_index_in_file = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					ω = 2π*ν_arr[ω_ind]

		    		G = read(Gsrc_file[1],robs_ind,:,1,1,ℓω_index_in_file)
		    		abs_α_robs² = G[1]^2 + G[2]^2

		    		ℓω_index_in_arr = get_index_in_split_array(all_modes,(ℓ,ω_ind))

		    		# m-averaged, so divided by 2ℓ+1
		    		Cℓω[ℓω_index_in_arr] = ω^2 * Powspec(ω) * 1/4π * abs_α_robs²

				end

				close(Gsrc_file)

			end

			return Cℓω

		end

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)

		futures = [@spawnat p summodes(rank) for (rank,p) in enumerate(procs)]

		Cℓω = zeros(ℓ_range,ν_ind_range)

		for f in futures
			Cℓω_part = fetch(f)
			for ind in axes(Cℓω_part,1)
				Cℓω[ind] = abs2(Cℓω_part[ind])
			end
		end

		return copy(transpose(Cℓω))
	end

	function Cωℓ_spectrum(ν::Real;kwargs...)
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr

		ν_test_ind = argmin(abs.(ν_arr .- ν))

		Cωℓ_spectrum(x1,x2;ν_ind_range=ν_test_ind:ν_test_ind,kwargs...)
	end

	########################################################################################################
	# Derivatives of cross-covariance
	########################################################################################################

	function ∂ϕ₂Cω(x1::Point3D,x2::Point3D;ℓ_range=nothing,ν_ind_range=nothing,r_src=Rsun-75e5)
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs ν_start_zeros Nν

		Nν_Gfn = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = max(ℓ_arr[1],ℓ_range[1]):min(ℓ_arr[end],ℓ_range[end])
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		end

		function Cω_summodes(rank)

			modes_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank) |> collect
			proc_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,modes_on_proc,num_procs)

			ν_ind_min = first(modes_on_proc)[2]
			ν_ind_max = last(modes_on_proc)[2]
			Nν_on_proc = ν_ind_max - ν_ind_min + 1

			Cω_proc = zeros(ComplexF64,ν_ind_min:ν_ind_max)

			∂ϕ₂Pl_cosχ = dPl(cosχ(x1,x2),ℓmax=ℓ_range[end]) .* ∂ϕ₂cosχ(x1,x2)

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind = argmin(abs.(r .- x2.r))

			for proc_id in proc_range
				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

		    	for (ℓ,ω_ind) in intersect(modes_on_proc,modes_in_file)
					
					mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					@assert(!isnothing(mode_index),"mode index $mode_index for ($ℓ,$ω_ind) not found by splitting across processors")

		    		ω = 2π * ν_arr[ω_ind]
		    		
		    		G = read(Gsrc_file[1],r₁_ind,:,1,1,mode_index)
		    		α_r₁ = G[1] + im*G[2]

		    		if r₁_ind == r₂_ind
		    			α_r₂ = α_r₁
		    		else
						G = read(Gsrc_file[1],r₂_ind,:,1,1,mode_index)
			    		α_r₂ = G[1] + im*G[2]
		    		end
		    		Cω_proc[ω_ind] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α_r₁) * α_r₂ * ∂ϕ₂Pl_cosχ[ℓ]
				end

				close(Gsrc_file)

			end

			return Cω_proc
		end

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		futures = [@spawnat p Cω_summodes(rank) for (rank,p) in enumerate(procs)]

		Cω_arr = zeros(ComplexF64,Nν)

		for f in futures
			Ci = fetch(f)
			ax = axes(Ci,1)
			@. Cω_arr[ax + ν_start_zeros] += Ci.parent
		end

		return Cω_arr
	end

	function ∂ϕ₂Cω(x1::Point3D,x2_arr::Vector{Point3D};ℓ_range=nothing,ν_ind_range=nothing,r_src=Rsun-75e5)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs ν_start_zeros Nν

		Nν_Gfn = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = intersect(ℓ_range,ℓ_arr)
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(ν_ind_range,1:Nν_Gfn)
		end

		∂ϕ₂Pl_cosχ = [dPl(cosχ(x1,x2),ℓmax=ℓ_range[end]) .* ∂ϕ₂cosχ(x1,x2) for x2 in x2_arr]

		function Cω_summodes(rank)

			modes_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank) |> collect
			proc_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,modes_on_proc,num_procs)

			ν_ind_min = first(modes_on_proc)[2]
			ν_ind_max = last(modes_on_proc)[2]
			Nν_on_proc = ν_ind_max - ν_ind_min + 1

			Cω_proc = zeros(ComplexF64,1:length(x2_arr),ν_ind_min:ν_ind_max)

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind_prev = argmin(abs.(r .- x2_arr[1].r))

			for proc_id in proc_range
				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

		    	for (ℓ,ω_ind) in intersect(modes_on_proc,modes_in_file)
					
					ℓω_index_in_file = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

		    		ω = 2π * ν_arr[ω_ind]
		    		
		    		G = read(Gsrc_file[1],r₁_ind,:,1,1,ℓω_index_in_file)
		    		α_r₁ = G[1] + im*G[2]

		    		for (x2ind,x2) in enumerate(x2_arr)
		    			r₂_ind = argmin(abs.(r .- x2.r))

			    		if x2ind==1 || r₂_ind != r₂_ind_prev
			    			if r₂_ind == r₁_ind
			    				α_r₂ = α_r₁
			    			else
				    			G = read(Gsrc_file[1],r₂_ind,:,1,1,ℓω_index_in_file)
				    			α_r₂ = G[1] + im*G[2]
				    		end
			    		end
		    			Cω_proc[x2ind,ω_ind] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α_r₁) * α_r₂ * ∂ϕ₂Pl_cosχ[x2ind][ℓ]
		    		end
				end

				close(Gsrc_file)

			end

			return Cω_proc
		end

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		futures = [@spawnat p Cω_summodes(rank) for (rank,p) in enumerate(procs)]

		Cω_arr = zeros(ComplexF64,length(x2_arr),Nν)

		for f in futures
			Ci = fetch(f)
			ν_range = axes(Ci,2)
			@. Cω_arr[:,ν_range + ν_start_zeros] += Ci.parent
		end

		return copy(transpose(Cω_arr))
	end

	function ∂ϕ₂Cω(n1::Point2D,n2_arr::Vector{Point2D};ℓ_range=nothing,ν_ind_range=nothing,r_src=Rsun-75e5,r_obs=Rsun-75e5)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs ν_start_zeros Nν

		Nν_Gfn = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = intersect(ℓ_range,ℓ_arr)
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(ν_ind_range,1:Nν_Gfn)
		end

		∂ϕ₂Pl_cosχ = [dPl(cosχ(n1,n2),ℓmax=ℓ_range[end]) .* ∂ϕ₂cosχ(n1,n2) for n2 in n2_arr]

		function Cω_summodes(rank)

			modes_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank) |> collect
			proc_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,modes_on_proc,num_procs)

			ν_ind_min = first(modes_on_proc)[2]
			ν_ind_max = last(modes_on_proc)[2]
			Nν_on_proc = ν_ind_max - ν_ind_min + 1

			Cω_proc = zeros(ComplexF64,1:length(n2_arr),ν_ind_min:ν_ind_max)

			robs_ind = argmin(abs.(r .- r_obs))

			for proc_id in proc_range
				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

		    	for (ℓ,ω_ind) in intersect(modes_on_proc,modes_in_file)
					
					ℓω_index_in_file = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

		    		ω = 2π * ν_arr[ω_ind]
		    		
		    		G = read(Gsrc_file[1],robs_ind,:,1,1,ℓω_index_in_file)
		    		αℓω2 = G[1]^2 + G[2]^2

		    		for (n2ind,n2) in enumerate(n2_arr)
		    			Cω_proc[n2ind,ω_ind] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * αℓω2 * ∂ϕ₂Pl_cosχ[n2ind][ℓ]
		    		end
				end

				close(Gsrc_file)

			end

			return Cω_proc
		end

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		futures = [@spawnat p Cω_summodes(rank) for (rank,p) in enumerate(procs)]

		Cω_arr = zeros(ComplexF64,length(n2_arr),Nν)

		for f in futures
			Ci = fetch(f)
			ν_range = axes(Ci,2)
			@. Cω_arr[:,ν_range + ν_start_zeros] += Ci.parent
		end

		return copy(transpose(Cω_arr))
	end
	########################################################################################################
	# Time-domain cross-covariance
	########################################################################################################

	function Ct(x1::Point3D,x2::Point3D;r_src=Rsun-75e5,kwargs...)
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dν Nt

		C = Cω(x1,x2;kwargs...)
		brfft(C,Nt,1) .* dν
	end

	Ct(n1::T,n2_arr::Vector{T};kwargs...) where T<:SphericalPoint = Ct(n1,n2;kwargs...)

	function ∂ϕ₂Ct(x1::Point3D,x2::Point3D;r_src=Rsun-75e5,kwargs...)
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dν Nt

		C = ∂ϕ₂Cω(x1,x2;kwargs...)
		brfft(C,Nt,1) .*dν
	end

	∂ϕ₂Ct(n1::T,n2_arr::Vector{T};kwargs...) where T<:SphericalPoint = ∂ϕ₂Ct(n1,n2;kwargs...)

	########################################################################################################
	# Cross-covariance at all distances on the equator, essentially the time-distance diagram
	########################################################################################################

	function CΔϕω(r₁::Real=Rsun-75e5,r₂::Real=Rsun-75e5;ℓ_range=nothing,r_src=Rsun-75e5,Δϕ_arr=nothing)

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs ν_full ν_start_zeros

		Nν_Gfn = length(ν_arr)

		r₁_ind = argmin(abs.(r .- r₁))
		r₂_ind = argmin(abs.(r .- r₂))

		if !isnothing(ℓ_range)
			ℓ_range = intersect(ℓ_range,ℓ_arr)
		else
			ℓ_range = ℓ_arr
		end

		ℓmax = ℓ_range[end]
		
		
		if isnothing(Δϕ_arr)
			nϕ = ℓmax
			Δϕ_arr = LinRange(0,π,nϕ+1)[1:end-1]
		else
			nϕ = length(Δϕ_arr)
		end

		Cϕω_arr = zeros(ComplexF64,nϕ,length(ν_full))

		Pl_cosχ = OffsetArray(zeros(ℓmax+1,nϕ),(-1,0))
		
		for (ϕ_ind,Δϕ) in enumerate(Δϕ_arr)
			Pl_cosχ[:,ϕ_ind] = Pl(cos(Δϕ),ℓmax=ℓmax)
		end

		Pl_cosχ = collect(transpose(Pl_cosχ))

		function summodes(rank)
			modes_on_proc = split_product_across_processors(ℓ_range,1:Nν_Gfn,num_workers,rank)
			proc_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,modes_on_proc,num_procs)

			ν_min_ind = first(collect(modes_on_proc))[2]
			ν_max_ind = last(collect(modes_on_proc))[2]
			Nν_on_proc = ν_max_ind - ν_min_ind + 1

			Cϕω_arr = OffsetArray(zeros(ComplexF64,nϕ,Nν_on_proc),(1:nϕ,ν_min_ind:ν_max_ind))

			for proc_id in proc_range
				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")

		    	# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

		    	for (ℓ,ω_ind) in intersect(modes_on_proc,modes_in_file)
					
					mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

		    		ω = 2π * ν_arr[ω_ind]

		    		
		    		G = read(Gsrc_file[1],r₁_ind,:,1,1,mode_index)
		    		α_r₁ = G[1] + im*G[2]

		    		if r₁_ind == r₂_ind
		    			α_r₂ = α_r₁
		    		else
						G = read(Gsrc_file[1],r₂_ind,:,1,1,mode_index)
			    		α_r₂ = G[1] + im*G[2]
		    		end

		    		for ϕ_ind in 1:nϕ
				    	Cϕω_arr[ϕ_ind,ω_ind] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α_r₁) * α_r₂ * Pl_cosχ[ϕ_ind,ℓ]
				    end

				end

				close(Gsrc_file)
			end

			return Cϕω_arr
		end

		procs = workers_active(ℓ_range,1:Nν_Gfn)
		num_workers = length(procs)
	    futures = [@spawnat p summodes(rank) for (rank,p) in enumerate(procs)]

	    for f in futures
	    	Ci = fetch(f)
	    	ν_inds = axes(Ci,2)
	    	@. Cϕω_arr[:,ν_inds .+ ν_start_zeros] += Ci.parent
	    end

		return Cϕω_arr
	end

	function CtΔϕ(r₁::Real=Rsun-75e5,r₂::Real=Rsun-75e5;r_src=Rsun-75e5,τ_ind_arr=nothing,kwargs...) 

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dν Nt

		CωΔϕ = copy(transpose(CΔϕω(r₁,r₂;r_src=r_src,kwargs...)))

		C = brfft(CωΔϕ,Nt,1).*dν

		if isnothing(τ_ind_arr)
			return C
		else
			return C[τ_ind_arr,:]
		end
	
		return 
	end

	function CΔϕt(r₁::Real=Rsun-75e5,r₂::Real=Rsun-75e5;r_src=Rsun-75e5,τ_ind_arr=nothing,kwargs...) 

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dν Nt

		CωΔϕ = copy(transpose(CΔϕω(r₁,r₂;r_src=r_src,kwargs...)))

		if isnothing(τ_ind_arr)
			τ_ind_arr = 1:Nt
		end
	
		return copy(transpose(brfft(CωΔϕ,Nt,1).*dν))[:,τ_ind_arr]
	end

	Cmω(r₁::Real=Rsun-75e5,r₂::Real=Rsun-75e5;kwargs...) = fft(CΔϕω(r₁,r₂;kwargs...),1)

	########################################################################################################
	# Cross-covariance in a rotating frame
	########################################################################################################

	function Cτ_rotating(x1::Point3D,x2::Point3D;Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,ℓ_range=nothing,r_src=Rsun-75e5)
		
		# Return C(Δϕ,ω) = RFFT(C(Δϕ,τ)) = RFFT(IRFFT(C0(Δϕ - Ωτ,ω))(τ))

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_full Nt dt

		if isnothing(τ_ind_arr)
			τ_ind_arr = 1:div(Nt,2)
		end

		Nτ = length(τ_ind_arr)

		Cτ_arr = OffsetArray{Float64}(undef,τ_ind_arr)
	    
		for τ_ind in τ_ind_arr
			τ = (τ_ind-1) * dt
	    	x2′ = Point3D(x2.r,x2.θ,x2.ϕ-Ω_rot*τ)
	    	Cτ_arr[τ_ind] = Ct(x1,x2′,ℓ_range=ℓ_range,r_src=r_src)[τ_ind]
	    end

		return Cτ_arr
	end

	function Cτ_rotating(x1::T,x2_arr::Vector{T};
		Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,ℓ_range=nothing,r_src=Rsun-75e5,kwargs...) where T<:SphericalPoint
		
		# Return C(Δϕ,ω) = RFFT(C(Δϕ,τ)) = RFFT(IRFFT(C0(Δϕ - Ωτ,ω))(τ))

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_full Nt dt

		if isnothing(τ_ind_arr)
			τ_ind_arr = [1:div(Nt,2) for x2 in x2_arr]
		end

		τ_ind_max_span = minimum(minimum.(τ_ind_arr)):maximum(maximum.(τ_ind_arr))

		Cτ_arr = [OffsetArray{Float64}(undef,τ_inds) for τ_inds in τ_ind_arr]
	    
		for τ_ind in τ_ind_max_span
			τ = (τ_ind-1) * dt
			if T==Point3D
		    	x2′_arr = [T(x2.r,x2.θ,x2.ϕ-Ω_rot*τ) for (rank,x2) in enumerate(x2_arr)
		    				 if τ_ind in τ_ind_arr[rank]]
		    elseif T==Point2D
		    	x2′_arr = [T(x2.θ,x2.ϕ-Ω_rot*τ) for (rank,x2) in enumerate(x2_arr)
		    				 if τ_ind in τ_ind_arr[rank]]
		    end

	    	x2′inds_arr = [rank for (rank,x2) in enumerate(x2_arr) if τ_ind in τ_ind_arr[rank]]
	    	Ct_x2_arr = Ct(x1,x2′_arr;ℓ_range=ℓ_range,r_src=r_src,kwargs...)[τ_ind,:]
	    	
	    	for (Ct_x2,x2ind) in zip(Ct_x2_arr,x2′inds_arr)
	    		Cτ_arr[x2ind][τ_ind] = Ct_x2
	    	end
	    end

		return Cτ_arr
	end

	#######################################################################################################################################

	function δCω_uniform_rotation_firstborn_integrated_over_angle(x1::Point3D,x2::Point3D,ν::Real;r_src=Rsun-75e5,kwargs...)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs

		Nν_Gfn = length(ν_arr)
		ν_test_ind = argmin(abs.(ν_arr .- ν))

		δCω_uniform_rotation_firstborn_integrated_over_angle(x1,x2;r_src=r_src,ν_ind_range=ν_test_ind:ν_test_ind,kwargs...)

	end

	function δCω_uniform_rotation_firstborn_integrated_over_angle(x1::Point3D,x2::Point3D;Ω_rot = 20e2/Rsun,ℓ_range=nothing,ν_ind_range=nothing,r_src=Rsun-75e5)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_x1 = Gfn_path_from_source_radius(x1)
		Gfn_path_x2 = Gfn_path_from_source_radius(x2)

		r₁_ind = argmin(abs.(r .- x1.r))
		r₂_ind = argmin(abs.(r .- x2.r))

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs
		Nν_Gfn = length(ν_arr)

		num_procs_obs1 = load(joinpath(Gfn_path_x1,"parameters.jld2"),"num_procs")
		num_procs_obs2 = load(joinpath(Gfn_path_x2,"parameters.jld2"),"num_procs")

		if !isnothing(ℓ_range)
			ℓ_range = intersect(ℓ_arr,ℓ_range)
		else
			ℓ_range = ℓ_arr
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(ν_ind_range,1:Nν_Gfn)
		end

		ℓmax = ℓ_arr[end]

		∂ϕ₂Pl_cosχ = dPl(cosχ(x1,x2),ℓmax=ℓmax) .*∂ϕ₂cosχ(x1,x2)

		# Gfn_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_path,ν,ν_arr,ℓ_arr,num_procs)

		δC = zeros(ComplexF64,Nν_Gfn)

		function δC_summodes(rank)
			modes_on_proc = collect(split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank))

			proc_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,modes_on_proc,num_procs)

			ν_min_ind = first(collect(modes_on_proc))[2]
			ν_max_ind = last(collect(modes_on_proc))[2]
			Nν_on_proc = ν_max_ind - ν_min_ind + 1

			δC_r = OffsetArray(zeros(ComplexF64,nr,Nν_on_proc),1:nr,ν_min_ind:ν_max_ind)

			δG_r₁_rsrc = zeros(ComplexF64,nr)

			if r₁_ind != r₂_ind
				δG_r₂_rsrc = zeros(ComplexF64,nr)
			else
				δG_r₂_rsrc = view(δG_r₁_rsrc,:)
			end

			for proc_id in proc_range

				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")

				# Get a list of (ℓ,ω) in Gsrc file
				modes_in_Gsrc_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(modes_on_proc,modes_in_Gsrc_file)
					
					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_Gsrc_file,(ℓ,ω_ind))
					@assert(!isnothing(ℓω_index),"mode index $ℓω_index for ($ℓ,$ω_ind) not found by splitting across processors")

		    		ω = 2π*ν_arr[ω_ind]
		    		
		    		G = read(Gsrc_file[1],:,:,1:2,1,ℓω_index_Gsrc_file)
		    		Gsrc_0 = G[:,1,1] + im*G[:,2,1]
		    		Gsrc_1 = G[:,1,2] + im*G[:,2,2]
		    		G_r₁_rsrc = Gsrc_0[r₁_ind]
		    		G_r₂_rsrc = Gsrc_0[r₂_ind]


		    		proc_id_mode_Gobs1,ℓω_index_Gobs1_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs1)
					Gobs1_file = FITS(joinpath(Gfn_path_x1,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs1),"r")

		    		G = read(Gobs1_file[1],:,:,1:2,1,ℓω_index_Gobs1_file)
		    		Gobs1_0 = G[:,1,1] + im*G[:,2,1]
		    		Gobs1_1 = G[:,1,2] + im*G[:,2,2]

		    		@. δG_r₁_rsrc = Gsrc_0 * Gobs1_0 - Gsrc_0 * Gobs1_1/Ω(ℓ,0) - Gsrc_1 * Gobs1_0/Ω(ℓ,0) +
				    								 (ℓ*(ℓ+1)-1) * Gsrc_1 * Gobs1_1/Ω(ℓ,0)^2

				    Gobs1_0 = nothing; Gobs1_1 = nothing;
				    close(Gobs1_file)

		    		if r₁_ind != r₂_ind

		    			proc_id_mode_Gobs2,ℓω_index_Gobs2_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs2)

						Gobs2_file = FITS(joinpath(Gfn_path_x2,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs2),"r")

			    		G = read(Gobs2_file[1],:,:,1:2,1,ℓω_index_Gobs2_file)
			    		Gobs2_0 = G[:,1,1] + im*G[:,2,1]
			    		Gobs2_1 = G[:,1,2] + im*G[:,2,2]

			    		@. δG_r₂_rsrc = Gsrc_0 * Gobs2_0 - Gsrc_0 * Gobs2_1/Ω(ℓ,0) - Gsrc_1 * Gobs2_0/Ω(ℓ,0) +
					    								 (ℓ*(ℓ+1)-1) * Gsrc_1 * Gobs2_1/Ω(ℓ,0)^2

					    Gobs2_0 = nothing; Gobs2_1 = nothing;
					    close(Gobs2_file)
					
					end

					@. δC_r[:,ω_ind] += ω^3*Powspec(ω)* (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] * (conj(G_r₁_rsrc) * δG_r₂_rsrc + conj(δG_r₁_rsrc) * G_r₂_rsrc) 

				end

				close(Gsrc_file)

			end

			return OffsetArray(integrate.simps((@. r^2 * ρ * δC_r.parent),x=r,axis=0),axes(δC_r,2))

		end

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		futures = [@spawnat p δC_summodes(rank) for (rank,p) in enumerate(procs)]

		for f in futures
			δC_i = fetch(f)
			ax = axes(δC_i,1)
			@. δC[ax] += δC_i
		end

		return @. -2im*Ω_rot*δC
	end

	########################################################################################################################################


	function δCω_uniform_rotation_rotatedwaves_linearapprox(x1::Point3D,x2::Point3D,ν::Real;Ω_rot= 20e2/Rsun,ℓ_range=nothing,r_src=Rsun-75e5)
		
		# We compute δC(x1,x2) = -iΩ ∑ℓ (2ℓ+1)/4π ∂ω (ω^2 P(ω) αℓω*(r₂)αℓω(r₁))∂Δϕ Pl(cos(Δϕ))
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs dν ν_start_zeros

		if !isnothing(ℓ_range)
			ℓ_range = intersect(ℓ_range,ℓ_arr)
		else
			ℓ_range = ℓ_arr
		end

		ν_test_ind = argmin(abs.(ν_arr .- ν))
		ν_on_grid = ν_arr[ν_test_ind]

		# @printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_range[1] ℓ_range[end]

		ν_ind_range = max(ν_test_ind-7,1):(ν_test_ind+min(7,ν_test_ind-1))
		ν_match_index = ν_test_ind - ν_ind_range[1] + 1
		dω = 2π*dν

		∂ϕC = ∂ϕ₂Cω(x1,x2,ν_ind_range=ν_ind_range,ℓ_range=ℓ_range,r_src=r_src)[ν_ind_range .+ ν_start_zeros]

	    ∂ω∂ϕC = D(length(∂ϕC))*∂ϕC ./ dω

		return -im*Ω_rot*∂ω∂ϕC[ν_match_index]
	end

	function δCω_uniform_rotation_rotatedwaves_linearapprox(x1::Point3D,x2::Point3D;Ω_rot= 20e2/Rsun,ℓ_range=nothing,ν_ind_range=nothing,r_src=Rsun-75e5)
		
		# We compute δC(x1,x2) = -iΩ ∑ℓ (2ℓ+1)/4π ∂ω (ω^2 P(ω) αℓω*(r₂)αℓω(r₁))∂Δϕ Pl(cos(Δϕ))
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr dν

		Nν_Gfn = length(ν_arr)

		if !isnothing(ℓ_range)
			ℓ_range = intersect(ℓ_range,ℓ_arr)
		else
			ℓ_range = ℓ_arr
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(ν_ind_range,1:Nν_Gfn)
		end
	
		dω = 2π*dν

		∂ϕC = ∂ϕ₂Cω(x1,x2,ℓ_range=ℓ_range,ν_ind_range=ν_ind_range,r_src=r_src)

	    ∂ω∂ϕC = D(length(∂ϕC))*∂ϕC ./ dω

		return @. -im*Ω_rot*∂ω∂ϕC
	end

	# INCOMPLETE!
	function δCω_uniform_rotation_rotatedwaves(Δϕ::Real,ν=3e-3;Ω_rot= 20e2/Rsun,ℓ_range=nothing,r_src=Rsun-75e5)
		# We compute δC(Δϕ,ω) = C′(Δϕ,ω) -  C(Δϕ,ω)
		C′(Δϕ,ω) = ifft(C(m,ω+mΩ),1)
		C0ωm_arr = collect(transpose(Cmω(;ℓ_range=ℓ_range)))
		C′ωm_arr = zeros(ComplexF64,size(C0ωm_arr))

		nm = size(C0ωm_arr,2); nϕ = nm
		m_arr = [0:div(nm,2)-1;-div(nm,2):1:-1]

		Δϕ_arr = LinRange(0,2π,nϕ+1)[1:end-1]

		# Find closest match
		Δϕ_ind = argmin(abs.(Δϕ_arr .- Δϕ))
		println("Using Δϕ = $(Δϕ_arr[Δϕ_ind]) instead of $(Δϕ)")

		Gfn_path = Gfn_path_from_source_radius(Rsun-75e5)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ν_full ν_start_zeros
		ω_arr_full = 2π .* ν_full
		ω_arr = 2π .* ν_arr

		Nν = length(ν_arr)

		ν_test_ind = argmin(abs.(ν_full .- ν))

		function interpolate_over_mω(rank)
			
			modes_on_proc = Main.parallel_utilities.split_product_across_processors(1:Nν,1:nm,nworkers(),rank)

			C′ωm_arr = zeros(ComplexF64,length(modes_on_proc))

			m_ind_prev = first(modes_on_proc)[2]
			Cω_interp_fn = interpolate((ω_arr_full,), C0ωm_arr[:,m_ind_prev], Gridded(Linear()))	

			for (mode_ind,(ω_ind,m_ind)) in enumerate(modes_on_proc)
				m = m_arr[m_ind]; ω = ω_arr[ω_ind]

				if m_ind != m_ind_prev
					# update the interpolation function
					m_ind_prev = m_ind
					Cω_interp_fn = interpolate((ω_arr_full,), C0ωm_arr[:,m_ind_prev], Gridded(Linear()))
				end
				
				
				C′ωm_arr[mode_ind] =  Cω_interp_fn(ω + m*Ω_rot)
				
			end
			return modes_on_proc,C′ωm_arr
		end

		futures = [@spawnat p interpolate_over_mω(rank) for (rank,p) in enumerate(workers())]

		for f in futures
			modes,C′ = fetch(f)
			for (mode_ind,(ν_ind,m_ind)) in enumerate(modes)
				C′ωm_arr[ν_ind+ν_start_zeros,m_ind] = C′[mode_ind]
			end
		end

		println(maximum(abs.(C0ωm_arr)))
		println(maximum(abs.(C′ωm_arr)))
		println(maximum(abs.(C′ωm_arr .- C0ωm_arr)))

		δCωϕ_arr = ifft(C′ωm_arr .- C0ωm_arr,2)

		δC_Δϕobs_νobs = δCωϕ_arr[ν_test_ind,Δϕ_ind]
	end

	#######################################################################################################################

	function δCt_uniform_rotation_rotatedwaves(x1::Point3D,x2::Point3D;Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,ℓ_range=nothing,r_src=Rsun-75e5)
		C′_t = Cτ_rotating(x1,x2,Ω_rot=Ω_rot,τ_ind_arr=τ_ind_arr,ℓ_range=ℓ_range,r_src=r_src)
		τ_ind_arr = axes(C′_t,1)
		C0_t = Ct(x1,x2,ℓ_range=ℓ_range,r_src=r_src)[τ_ind_arr]
		return C′_t .- C0_t
	end

	function δCt_uniform_rotation_rotatedwaves(x1::T,x2_arr::Vector{T};
		Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,kwargs...) where T<:SphericalPoint

		C′_t = Cτ_rotating(x1,x2_arr;Ω_rot=Ω_rot,τ_ind_arr=τ_ind_arr,kwargs...)
		τ_ind_arr = [axes(Cx2,1) for Cx2 in C′_t]
		C0_t = [Ct(x1,x2_arr;kwargs...)[τ_ind,ind] for (ind,τ_ind) in enumerate(τ_ind_arr)]
		return C′_t .- C0_t
	end

	function δCt_uniform_rotation_rotatedwaves_linearapprox(x1::Point3D,x2::Point3D;Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,ℓ_range=ℓ_range,r_src=Rsun-75e5)
		Gfn_path = Gfn_path_from_source_radius(x1)
		@load "$Gfn_path/parameters.jld2" Nt dt
		
		t = (0:Nt-1).*dt
		
		δCt = -Ω_rot .* t .* ∂ϕ₂Ct(x1,x2,ℓ_range=ℓ_range,r_src=r_src)
		if !isnothing(τ_ind_arr)
			return δCt[τ_ind_arr]
		else
			return δCt
		end
	end	

	function δCt_uniform_rotation_rotatedwaves_linearapprox(x1::Point3D,x2_arr::Vector{Point3D};Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,ℓ_range=ℓ_range,r_src=Rsun-75e5)
		Gfn_path = Gfn_path_from_source_radius(x1)
		@load "$Gfn_path/parameters.jld2" Nt dt
		
		t = (0:Nt-1).*dt
		
		δCt = -Ω_rot .* t .* ∂ϕ₂Ct(x1,x2_arr,ℓ_range=ℓ_range,r_src=r_src)
		δCt_arr = Vector{Float64}[]
		if !isnothing(τ_ind_arr)
			for x2ind in 1:length(x2_arr)
				push!(δCt_arr,δCt[τ_ind_arr[x2ind],x2ind])
			end
		else
			δCt_arr = δCt
		end
		return δCt_arr
	end

	function δCt_uniform_rotation_rotatedwaves_linearapprox(n1::Point2D,n2_arr::Vector{Point2D};Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,r_src=Rsun-75e5,kwargs...)
		Gfn_path = Gfn_path_from_source_radius(r_src)
		@load "$Gfn_path/parameters.jld2" Nt dt
		
		t = (0:Nt-1).*dt
		
		δCt = -Ω_rot .* t .* ∂ϕ₂Ct(n1,n2_arr;r_src=r_src,kwargs...)
		δCt_arr = Vector{Float64}[]
		if !isnothing(τ_ind_arr)
			for (n2ind,τ_inds) in enumerate(τ_ind_arr)
				push!(δCt_arr,δCt[τ_inds,n2ind])
			end
		else
			δCt_arr = δCt
		end
		return δCt_arr
	end

	########################################################################################################################

	function bounce_filter(Δϕ,n)
		nparams = 5
		coeffs = Dict()
		for i in [1,2,4]
			coeffs[i] = Dict("high"=>zeros(nparams),"low"=>zeros(nparams))
		end

		coeffs[1]["high"] = [2509.132334896018,12792.508296270391,-13946.527195127102,8153.75242742649,-1825.7567469552703]
		coeffs[1]["low"] = [40.821191938380714,11410.21390421857,-11116.305124138207,5254.244817703224,-895.0009393800744]

		coeffs[2]["high"] = [4083.6946001848364,14924.442447995087,-13880.238239469609,7562.499279468063,-1622.5318939228978]
		coeffs[2]["low"] = [2609.4406668522433,10536.81683213881,-7023.811081076518,2586.7238222832298,-348.9245124332354]

		coeffs[4]["high"] = [6523.103468645263,16081.024611219753,-7085.7174198723405,973.4990690666436,236.95568587146957]
		coeffs[4]["low"] = [5150.314633252216,15040.045600508669,-8813.047362534506,3878.5398150601663,-870.3633232120256]

		τ_low,τ_high = 0.,0.
		for (i,c) in enumerate(coeffs[n]["high"])
			τ_high += c*Δϕ^(i-1)
		end

		for (i,c) in enumerate(coeffs[n]["low"])
			τ_low += c*Δϕ^(i-1)
		end

		return τ_low,τ_high
	end

	function gaussian_fit(x,y)
		# use the fact that if y=Gaussian(x), log(y) = quadratic(x)
		# quadratic(x) = ax² + bx + c
		# Gaussian(x) = A*exp(-(x-x0)²/2σ²)
		# the parameters are σ=√(-1/2a), x0 = -b/2a, A=exp(c-b^2/4a)
		c,b,a=polyfit(x,log.(y),2).a
		A = exp(c-b^2/4a)
		x0 = -b/2a
		σ = √(-1/2a)
		return A,x0,σ
	end

	function h(x1::Point3D,x2::Point3D;plots=false,bounce_no=1,ℓ_range=nothing,r_src=Rsun-75e5)

		Gfn_path_x1 = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path_x1,"parameters.jld2") ν_full ν_start_zeros ν_arr Nt dt dν

		ω_full = 2π.*ν_full

		Cω_x1x2 = Cω(x1,x2,ℓ_range=ℓ_range,r_src=r_src)

		C_t = brfft(Cω_x1x2,Nt).*dν

		env = abs.(hilbert(C_t))[1:div(Nt,2)] # first half for positive shifts

		# get the time of first bounce arrival in seconds
		τ_low,τ_high = bounce_filter(acos(cosχ(x1,x2)),bounce_no)
		τ_low_ind = Int(floor(τ_low/dt)); τ_high_ind = Int(ceil(τ_high/dt))

		# fit a gaussian to obtain the correct bounce
		peak_center = argmax(env[τ_low_ind:τ_high_ind]) + τ_low_ind - 1
		points_around_max = env[peak_center-2:peak_center+2]
		amp = env[peak_center]
		# println("$amp $peak_center")
		# println(points_around_max)
		# Assuming a roughly gaussian peak, fit a quadratic to log(Cω) to obtain σ

		A,t0,σt = gaussian_fit(peak_center-2:peak_center+2, points_around_max)

		t_inds_range = Int(floor(t0 - 2σt)):Int(ceil(t0 + 2σt))
		f_t = zeros(Nt)
		f_t[t_inds_range] .= 1

		∂tCt = brfft(Cω_x1x2.*im.*ω_full,Nt).*dν
		h_t = f_t .* ∂tCt ./ sum(f_t.*∂tCt.^2 .* dt)
		h_ω = rfft(h_t) .* dt

		if plots
			plt.subplot(411)
			plt.plot(ν_arr,real(Cω_x1x2[ν_start_zeros .+ (1:length(ν_arr))]))
			plt.title("C(x₁,x₂,ω)")

			ax2=plt.subplot(412)
			plt.plot((1:Nt).*dt,C_t,color="black")
			plt.axvline(t_inds_range[1]*dt,ls="solid")
			plt.axvline(t_inds_range[end]*dt,ls="solid")
			# plt.xlim(0,60^2*6)
			plt.title("C(x₁,x₂,t)")

			plt.subplot(413,sharex=ax2)
			plt.plot((1:Nt).*dt,h_t,color="black")
			plt.xlim(0,60^2*6)

			plt.title("h(x₁,x₂,t)")

			plt.subplot(414)
			plt.plot(ν_arr,imag(h_ω[ν_start_zeros .+ (1:length(ν_arr))]),label="imag")
			plt.plot(ν_arr,real(h_ω[ν_start_zeros .+ (1:length(ν_arr))]),label="real")
			plt.legend(loc="best")
			plt.title("h(x₁,x₂,ω)")

			plt.tight_layout()
		end
		
		return (t_inds_range=t_inds_range,f_t=f_t,h_t=h_t,h_ω=h_ω)
	end

	##########################################################################################################################
	# Add methods for computing cross-covariances in 2D (same obs radii) and 1D (same obs radii and on the equator)
	##########################################################################################################################

	for fn in (:∂ϕ₂Cω,:∂ϕ₂Ct,:Ct,:Cτ_rotating,
		:δCω_uniform_rotation_firstborn_integrated_over_angle,:δCω_uniform_rotation_firstborn_integrated_over_angle,
		:δCω_uniform_rotation_rotatedwaves_linearapprox,:δCω_uniform_rotation_rotatedwaves_linearapprox,
		:δCt_uniform_rotation_rotatedwaves,:δCt_uniform_rotation_rotatedwaves_linearapprox,:h)

		@eval $fn(n1::Point2D,n2::Point2D;r_obs::Real=Rsun-75e5,kwargs...) = $fn(Point3D(r_obs,n1),Point3D(r_obs,n2);kwargs...)
		@eval $fn(Δϕ::Real;r_obs::Real=Rsun-75e5,kwargs...) = $fn(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ);kwargs...)
	end

end

#################################################################
# Kernels and travel times
#################################################################

module kernel

	using Reexport
	@reexport using Main.crosscov
	@pyimport scipy.integrate as integrate
	import WignerSymbols: clebschgordan
	using WignerD
	using FileIO
	using Profile

	################################################################################################################
	# Validation for uniform rotation
	################################################################################################################

	uniform_rotation_uplus(Ω_rot=20e2/Rsun) = @. √(4π/3)*im*Ω_rot*r

	function kernel_uniform_rotation_uplus(x1::Point3D,x2::Point3D;ℓ_range=nothing,ν_ind_range=nothing,bounce_no=1,r_src=Rsun-75e5)
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_x1 = Gfn_path_from_source_radius(x1)
		Gfn_path_x2 = Gfn_path_from_source_radius(x2)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs dν ν_start_zeros

		num_procs_obs1 = load(joinpath(Gfn_path_x1,"parameters.jld2"),"num_procs")
		num_procs_obs2 = load(joinpath(Gfn_path_x2,"parameters.jld2"),"num_procs")

		Nν_Gfn = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = intersect(ℓ_range,ℓ_arr)
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(1:Nν_Gfn,ν_ind_range)
		end

		h_ω_arr = h(x1,x2,bounce_no=bounce_no,ℓ_range=ℓ_range,r_src=r_src).h_ω[ν_start_zeros .+ ν_ind_range] # only in range

		r₁_ind = argmin(abs.(r .- x1.r))
		r₂_ind = argmin(abs.(r .- x2.r))

		dω = dν*2π		

		function sum_modes(rank)
		
			ℓ_ωind_iter_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank)
			proc_id_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)

			∂ϕ₂Pl_cosχ = dPl(cosχ(x1,x2),ℓmax=ℓ_arr[end]).*∂ϕ₂cosχ(x1,x2)

			K = zeros(ComplexF64,nr)

			δG_r₁_rsrc = zeros(ComplexF64,nr) # This is not actually δG, it's just a radial function dependent on f

			G = zeros(nr,2,2)
			Gsrc = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)
			Gobs1 = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)

			if r₁_ind != r₂_ind
				δG_r₂_rsrc = zeros(ComplexF64,nr)
				Gobs2 = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)
			else
				δG_r₂_rsrc = view(δG_r₁_rsrc,:)
			end

			for proc_id in proc_id_range

				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")				
				# Get a list of (ℓ,ω) in this file
				modes_in_Gsrc_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)
				

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_Gsrc_file)

					# Get index of this mode in the file
					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_Gsrc_file,(ℓ,ω_ind))
					ω = 2π*ν_arr[ω_ind]

					# Green function about source location
		    		G .= read(Gsrc_file[1],:,:,1:2,1,ℓω_index_Gsrc_file)
		    		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		G_r₁_rsrc = Gsrc[r₁_ind,0]
		    		G_r₂_rsrc = Gsrc[r₂_ind,0]

		    		# Green function about receiver location
		    		proc_id_mode_Gobs1,ℓω_index_Gobs1_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs1)
		    		Gobs1_file = FITS(joinpath(Gfn_path_x1,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs1),"r")

		    		G .= read(Gobs1_file[1],:,:,1:2,1,ℓω_index_Gobs1_file)
		    		@. Gobs1[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gobs1[:,1] = G[:,1,2] + im*G[:,2,2]

		    		@. δG_r₁_rsrc = Gsrc[:,0] * Gobs1[:,0] - Gsrc[:,0] * Gobs1[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs1[:,0] + (ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs1[:,1]/Ω(ℓ,0)

					close(Gobs1_file)

				    if r₁_ind != r₂_ind

				    	proc_id_mode_Gobs2,ℓω_index_Gobs2_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs2)

				    	Gobs2_file = FITS(joinpath(Gfn_path_x2,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs2),"r")
			    		G .= read(Gobs2_file[1],:,:,1:2,1,ℓω_index_Gobs2_file)
			    		@. Gobs2[:,0] = G[:,1,1] + im*G[:,2,1]
			    		@. Gobs2[:,1] = G[:,1,2] + im*G[:,2,2]

			    		@. δG_r₂_rsrc = Gsrc[:,0] * Gobs2[:,0] - Gsrc[:,0] * Gobs2[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs2[:,0] + (ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs2[:,1]/Ω(ℓ,0)

					    close(Gobs2_file)
					     
					end


					@. K +=  dω/2π * ω^3 * Powspec(ω) * (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] * imag(h_ω_arr[ω_ind]) * 
								(conj(δG_r₁_rsrc)*G_r₂_rsrc + conj(G_r₁_rsrc)*δG_r₂_rsrc) 

				end

				close(Gsrc_file)
			end

			return 4*√(3/4π)*im .* K .* r .* ρ
		end

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		kernel_futures = [@spawnat p sum_modes(rank) for (rank,p) in enumerate(procs)]

		K = sum(fetch.(kernel_futures))

		return K
	end

	function kernel_uniform_rotation_uplus(x1::Point3D,x2_arr::Vector{Point3D};ℓ_range=nothing,ν_ind_range=nothing,bounce_no=1,r_src=Rsun-75e5)
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_x1 = Gfn_path_from_source_radius(x1)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs dν ν_start_zeros

		num_procs_obs1 = load(joinpath(Gfn_path_x1,"parameters.jld2"),"num_procs")

		Nν_Gfn = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = intersect(ℓ_range,ℓ_arr)
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(1:Nν_Gfn,ν_ind_range)
		end

		r₁_ind = argmin(abs.(r .- x1.r))
		r₂_ind_prev = argmin(abs.(r .- x2_arr[1].r))

		dω = dν*2π

		function sum_modes(rank)
		
			ℓ_ωind_iter_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank)
			proc_id_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)

			K = zeros(ComplexF64,nr,length(x2_arr))

			δG_r₁_rsrc = zeros(ComplexF64,nr) # This is not actually δG, it's just a radial function dependent on f
			δG_r₂_rsrc = zeros(ComplexF64,nr)

			G = zeros(nr,2,2)
			Gsrc = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)
			Gobs1 = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)
			Gobs2 = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)			

			for proc_id in proc_id_range

				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")				
				# Get a list of (ℓ,ω) in this file
				modes_in_Gsrc_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_Gsrc_file)

					# Get index of this mode in the file
					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_Gsrc_file,(ℓ,ω_ind))
					ω = 2π*ν_arr[ω_ind]

					# Green function about source location
		    		G .= read(Gsrc_file[1],:,:,1:2,1,ℓω_index_Gsrc_file)
		    		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		G_r₁_rsrc = Gsrc[r₁_ind,0]
		    		G_r₂_rsrc = Gsrc[r₂_ind_prev,0]

		    		# Green function about receiver location
		    		proc_id_mode_Gobs1,ℓω_index_Gobs1_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs1)
		    		Gobs1_file = FITS(joinpath(Gfn_path_x1,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs1),"r")

		    		G .= read(Gobs1_file[1],:,:,1:2,1,ℓω_index_Gobs1_file)
		    		@. Gobs1[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gobs1[:,1] = G[:,1,2] + im*G[:,2,2]

		    		@. δG_r₁_rsrc = Gsrc[:,0] * Gobs1[:,0] - Gsrc[:,0] * Gobs1[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs1[:,0] + (ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs1[:,1]/Ω(ℓ,0)

					close(Gobs1_file)

					for (x2ind,x2) in enumerate(x2_arr)
						∂ϕ₂Pl_cosχ = dPl(cosχ(x1,x2),ℓmax=ℓ_arr[end]).*∂ϕ₂cosχ(x1,x2)
						r₂_ind = argmin(abs.(r .- x2.r))

					    if x2ind==1 || r₂_ind_prev != r₂_ind

					    	Gfn_path_x2 = Gfn_path_from_source_radius(x2)
					    	num_procs_obs2 = load(joinpath(Gfn_path_x2,"parameters.jld2"),"num_procs")
					    	G_r₂_rsrc = Gsrc[r₂_ind,0]

					    	proc_id_mode_Gobs2,ℓω_index_Gobs2_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs2)

					    	Gobs2_file = FITS(joinpath(Gfn_path_x2,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs2),"r")
				    		G .= read(Gobs2_file[1],:,:,1:2,1,ℓω_index_Gobs2_file)
				    		@. Gobs2[:,0] = G[:,1,1] + im*G[:,2,1]
				    		@. Gobs2[:,1] = G[:,1,2] + im*G[:,2,2]

				    		@. δG_r₂_rsrc = Gsrc[:,0] * Gobs2[:,0] - Gsrc[:,0] * Gobs2[:,1]/Ω(ℓ,0) -
			    						Gsrc[:,1]/Ω(ℓ,0) * Gobs2[:,0] + (ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs2[:,1]/Ω(ℓ,0)

						    close(Gobs2_file)
						     
						end

						h_ω_arr_x2 = h_ω_arr[x2ind]

						@. K[:,x2ind] +=  dω/2π * ω^3 * Powspec(ω) * (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] * imag(h_ω_arr_x2[ω_ind]) * 
									(conj(δG_r₁_rsrc)*G_r₂_rsrc + conj(G_r₁_rsrc)*δG_r₂_rsrc) 

					end

				end

				close(Gsrc_file)
			end

			return 4*√(3/4π)*im .* K .* r .* ρ
		end

		h_ω_arr = [h(x1,x2,bounce_no=bounce_no,ℓ_range=ℓ_range,r_src=r_src).h_ω[ν_start_zeros .+ ν_ind_range] for x2 in x2_arr]# only in range

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		kernel_futures = [@spawnat p sum_modes(rank) for (rank,p) in enumerate(procs)]

		K = sum(fetch.(kernel_futures))

		return K
	end

	function kernel_uniform_rotation_uplus(n1::Point2D,n2::Point2D;ℓ_range=nothing,ν_ind_range=nothing,bounce_no=1,r_src=Rsun-75e5,r_obs=Rsun-75e5)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_obs = Gfn_path_from_source_radius(r_obs)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs dν ν_start_zeros

		num_procs_obs = load(joinpath(Gfn_path_obs,"parameters.jld2"),"num_procs")

		Nν_Gfn = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = intersect(ℓ_range,ℓ_arr)
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(1:Nν_Gfn,ν_ind_range)
		end

		robs_ind = argmin(abs.(r .- r_obs))

		dω = dν*2π

		∂ϕ₂Pl_cosχ = dPl(cosχ(n1,n2),ℓmax=ℓ_arr[end]).*∂ϕ₂cosχ(n1,n2)
		h_ω_arr = h(n1,n2,bounce_no=bounce_no,ℓ_range=ℓ_range,r_src=r_src,r_obs=r_obs).h_ω[ν_start_zeros .+ ν_ind_range] # only in range

		function sum_modes(rank)
		
			ℓ_ωind_iter_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank)
			proc_id_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)

			K = zeros(ComplexF64,nr)

			δG_robs_rsrc = zeros(ComplexF64,nr) # This is not actually δG, it's just a radial function dependent on f

			G = zeros(nr,2,2)
			Gsrc = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)
			Gobs = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)

			for proc_id in proc_id_range

				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")				
				# Get a list of (ℓ,ω) in this file
				modes_in_Gsrc_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				# check if number of procs is the same for the source and observer, 
				# in this case the file indices are identical.
				if num_procs_obs == num_procs
					# Green function about receiver location
		    		Gobs_file = FITS(joinpath(Gfn_path_obs,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")
		    	end

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_Gsrc_file)

					# Get index of this mode in the file
					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_Gsrc_file,(ℓ,ω_ind))
					ω = 2π*ν_arr[ω_ind]

					# Green function about source location
		    		G .= read(Gsrc_file[1],:,:,1:2,1,ℓω_index_Gsrc_file)
		    		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		G_robs_rsrc = Gsrc[robs_ind,0]

		    		# Green function about receiver location
		    		if num_procs_obs != num_procs
		    			proc_id_mode_Gobs,ℓω_index_Gobs_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs)
		    			Gobs_file = FITS(joinpath(Gfn_path_obs,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs),"r")
		    		else
		    			ℓω_index_Gobs_file = ℓω_index_Gsrc_file
		    		end

		    		G .= read(Gobs_file[1],:,:,1:2,1,ℓω_index_Gobs_file)
		    		@. Gobs[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gobs[:,1] = G[:,1,2] + im*G[:,2,2]

		    		@. δG_robs_rsrc = Gsrc[:,0] * Gobs[:,0] - Gsrc[:,0] * Gobs[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs[:,0] + (ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs[:,1]/Ω(ℓ,0)

		    		if num_procs_obs != num_procs
						close(Gobs_file)
					end

					@. K +=  dω/2π * ω^3 * Powspec(ω) * (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] * imag(h_ω_arr[ω_ind]) * 
								2real(conj(δG_robs_rsrc)*G_robs_rsrc) 

				end

				close(Gsrc_file)
				if num_procs_obs == num_procs
					close(Gobs_file)
				end
			end

			return 4*√(3/4π)*im .* K .* r .* ρ
		end		

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		kernel_futures = [@spawnat p sum_modes(rank) for (rank,p) in enumerate(procs)]

		K = sum(fetch.(kernel_futures))

		return K
	end
	
	function kernel_uniform_rotation_uplus(n1::Point2D,n2_arr::Vector{Point2D};ℓ_range=nothing,ν_ind_range=nothing,bounce_no=1,r_src=Rsun-75e5,r_obs=Rsun-75e5)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_obs = Gfn_path_from_source_radius(r_obs)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs dν ν_start_zeros

		num_procs_obs = load(joinpath(Gfn_path_obs,"parameters.jld2"),"num_procs")

		Nν_Gfn = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = intersect(ℓ_range,ℓ_arr)
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(1:Nν_Gfn,ν_ind_range)
		end

		robs_ind = argmin(abs.(r .- r_obs))

		dω = dν*2π

		function sum_modes(rank)
		
			ℓ_ωind_iter_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank)
			proc_id_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)

			K = zeros(ComplexF64,nr,length(n2_arr))

			δG_robs_rsrc = zeros(ComplexF64,nr) # This is not actually δG, it's just a radial function dependent on f

			G = zeros(nr,2,2)
			Gsrc = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)
			Gobs = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)

			for proc_id in proc_id_range

				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")				
				# Get a list of (ℓ,ω) in this file
				modes_in_Gsrc_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				# check if number of procs is the same for the source and observer, 
				# in this case the file indices are identical.
				if num_procs_obs == num_procs
					# Green function about receiver location
		    		Gobs_file = FITS(joinpath(Gfn_path_obs,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")
		    	end

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_Gsrc_file)

					# Get index of this mode in the file
					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_Gsrc_file,(ℓ,ω_ind))
					ω = 2π*ν_arr[ω_ind]

					# Green function about source location
		    		G .= read(Gsrc_file[1],:,:,1:2,1,ℓω_index_Gsrc_file)
		    		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		G_robs_rsrc = Gsrc[robs_ind,0]

		    		# Green function about receiver location
		    		if num_procs_obs != num_procs
		    			proc_id_mode_Gobs,ℓω_index_Gobs_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs)
		    			Gobs_file = FITS(joinpath(Gfn_path_obs,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs),"r")
		    		else
		    			ℓω_index_Gobs_file = ℓω_index_Gsrc_file
		    		end

		    		G .= read(Gobs_file[1],:,:,1:2,1,ℓω_index_Gobs_file)
		    		@. Gobs[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gobs[:,1] = G[:,1,2] + im*G[:,2,2]

		    		@. δG_robs_rsrc = Gsrc[:,0] * Gobs[:,0] - Gsrc[:,0] * Gobs[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs[:,0] + (ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs[:,1]/Ω(ℓ,0)

		    		if num_procs_obs != num_procs
						close(Gobs_file)
					end

					for (n2ind,n2) in enumerate(n2_arr)
						∂ϕ₂Pl_cosχ = ∂ϕ₂Pl_cosχ_arr[n2ind]
						h_ω_arr_n2 = h_ω_arr[n2ind]

						@. K[:,n2ind] +=  dω/2π * ω^3 * Powspec(ω) * (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] * imag(h_ω_arr_n2[ω_ind]) * 
									2real(conj(δG_robs_rsrc)*G_robs_rsrc) 

					end

				end

				close(Gsrc_file)
				if num_procs_obs == num_procs
					close(Gobs_file)
				end
			end

			return 4*√(3/4π)*im .* K .* r .* ρ
		end

		∂ϕ₂Pl_cosχ_arr = [dPl(cosχ(n1,n2),ℓmax=ℓ_arr[end]).*∂ϕ₂cosχ(n1,n2) for n2 in n2_arr]
		h_ω_arr = [h(n1,n2,bounce_no=bounce_no,ℓ_range=ℓ_range,r_src=r_src,r_obs=r_obs).h_ω[ν_start_zeros .+ ν_ind_range] for n2 in n2_arr]# only in range

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		kernel_futures = [@spawnat p sum_modes(rank) for (rank,p) in enumerate(procs)]

		K = sum(fetch.(kernel_futures))

		return K
	end

	function δτ_uniform_rotation_firstborn_int_K_u(x1,x2;Ω_rot=20e2/Rsun,kwargs...)
		K₊ = kernel_uniform_rotation_uplus(x1,x2;kwargs...)
		u⁺ = uniform_rotation_uplus(Ω_rot)

		δτ = real.(integrate.simps(K₊.*u⁺,x=r,axis=0))
	end

	function δτ_uniform_rotation_firstborn_int_hω_δCω(n1::Point2D,n2::Point2D;r_src=Rsun-75e5,Ω_rot=20e2/Rsun,bounce_no=1,kwargs...)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr dν ν_start_zeros

		h_ω_arr = h(n1,n2;bounce_no=bounce_no,kwargs...).h_ω[ν_start_zeros .+ (1:length(ν_arr))] # only in range

		dω = dν*2π

		Nν = length(ν_arr)

		δτ = zero(Float64)

		δC = δCω_uniform_rotation_firstborn_integrated_over_angle(n1,n2;Ω_rot=Ω_rot,kwargs...)

		for (hω,δCω) in zip(h_ω_arr,δC)
			δτ += dω/2π * 2real(conj(hω)*δCω)
		end

		return δτ
	end	

	function δτ_uniform_rotation_rotatedframe_int_hω_δCω_linearapprox(n1::Point2D,n2::Point2D;r_src=Rsun-75e5,Ω_rot=20e2/Rsun,bounce_no=1,kwargs...)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr dν ν_start_zeros

		Nν_Gfn = length(ν_arr)
		ν_Gfn_ind_range = ν_start_zeros .+ (1:Nν_Gfn)

		h_ω_arr = h(n1,n2;bounce_no=bounce_no,kwargs...).h_ω[ν_Gfn_ind_range] # only in range

		dω = dν*2π

		δτ = zero(Float64)

		δC = δCω_uniform_rotation_rotatedwaves_linearapprox(n1,n2;Ω_rot=Ω_rot,kwargs...)[ν_Gfn_ind_range]

		for (hω,δCω) in zip(h_ω_arr,δC)
			δτ += dω/2π * 2real(conj(hω)*δCω)
		end

		return δτ
	end

	function δτ_uniform_rotation_rotatedframe_int_ht_δCt(n1::Point2D,n2::Point2D;r_src=Rsun-75e5,Ω_rot=20e2/Rsun,bounce_no=1,kwargs...)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt

		h_arr = h(n1,n2;bounce_no=bounce_no,kwargs...)
		h_t = h_arr.h_t
		τ_ind_arr = h_arr.t_inds_range

		δC_t = δCt_uniform_rotation_rotatedwaves(n1,n2;Ω_rot=Ω_rot,τ_ind_arr=τ_ind_arr,kwargs...)

		δτ = integrate.simps(h_t[τ_ind_arr].*δC_t.parent,dx=dt)
	end

	function δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1::Point2D,n2::Point2D;r_src=Rsun-75e5,Ω_rot=20e2/Rsun,bounce_no=1,kwargs...)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt

		h_arr = h(n1,n2;bounce_no=bounce_no,kwargs...)
		h_t = h_arr.h_t
		τ_ind_arr = h_arr.t_inds_range

		δC_t = δCt_uniform_rotation_rotatedwaves_linearapprox(n1,n2;Ω_rot=Ω_rot,τ_ind_arr=τ_ind_arr,kwargs...)

		δτ = integrate.simps(h_t[τ_ind_arr].*δC_t,dx=dt,axis=0)
	end

	function δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1::Point2D,n2_arr::Vector{Point2D};r_src=Rsun-75e5,Ω_rot=20e2/Rsun,bounce_no=1,kwargs...)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt

		δτ = zeros(length(n2_arr))

		h_arr = [h(n1,n2;bounce_no=bounce_no,kwargs...) for n2 in n2_arr]
		τ_ind_arr = [h_i.t_inds_range for h_i in h_arr]

		δC_t = δCt_uniform_rotation_rotatedwaves_linearapprox(n1,n2_arr;Ω_rot=Ω_rot,τ_ind_arr=τ_ind_arr,kwargs...)

		for (n2ind,τ_inds) in enumerate(τ_ind_arr)
			δτ[n2ind] = integrate.simps(h_arr[n2ind].h_t[τ_inds].*δC_t[n2ind],dx=dt,axis=0)
		end

		return δτ
	end

	function traveltimes_validate(n1,n2;kwargs...)

		δτ1 = δτ_uniform_rotation_firstborn_int_K_u(n1,n2;kwargs...)
		@printf "%-50s %g\n" "First Born, ∫dr u(r) K(r)" round(δτ1,sigdigits=3)

		δτ2 = δτ_uniform_rotation_firstborn_int_hω_δCω(n1,n2;kwargs...)
		@printf "%-50s %g\n" "First Born, ∫dω/2π h(ω) δC_FB(ω)" round(δτ2,sigdigits=3)

		δτ3 = δτ_uniform_rotation_rotatedframe_int_hω_δCω_linearapprox(n1,n2;kwargs...)
		@printf "%-50s %g\n" "Rotated frame, ∫dω/2π h(ω) δC_R_lin(ω)" round(δτ3,sigdigits=3)

		δτ4 = δτ_uniform_rotation_rotatedframe_int_ht_δCt(n1,n2;kwargs...)
		@printf "%-50s %g\n" "Rotated frame, ∫dt h(t) δC_R(t)" round(δτ4,sigdigits=3)

		δτ5 = δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1,n2;kwargs...)
		@printf "%-50s %g\n" "Rotated frame, ∫dt h(t) δC_R_lin(t)" round(δτ5,sigdigits=3)
	end

	################################################################################################################


	################################################################################################################
	# All kernels
	################################################################################################################
	
	Nℓ′ℓs(ℓ′,ℓ,s) = √((2ℓ+1)*(2ℓ′+1)/(4π*(2s+1)))

	function flow_kernels_srange_t0(x1::Point3D,x2::Point3D,s_max;ℓ_range=nothing,ν_ind_range=nothing,bounce_no=1,r_src=Rsun-75e5,K_components=-1:1)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_x1 = Gfn_path_from_source_radius(x1)
		Gfn_path_x2 = Gfn_path_from_source_radius(x2)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs dν ν_start_zeros

		num_procs_obs1 = load(joinpath(Gfn_path_x1,"parameters.jld2"),"num_procs")
		num_procs_obs2 = load(joinpath(Gfn_path_x2,"parameters.jld2"),"num_procs")

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = intersect(ℓ_range,ℓ_arr)
		end

		Nν_Gfn = length(ν_arr)

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν_Gfn
		else
			ν_ind_range = intersect(1:Nν_Gfn,ν_ind_range)
		end

		h_ω_arr = h(x1,x2,bounce_no=bounce_no,ℓ_range=ℓ_range,r_src=r_src).h_ω[ν_start_zeros .+ ν_ind_range] # only in range

		r₁_ind = argmin(abs.(r .- x1.r))
		r₂_ind = argmin(abs.(r .- x2.r))

		dω = dν*2π

		function sum_modes(rank)
		
			ℓ_ωind_iter_on_proc = collect(split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank))
			proc_id_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)

			K = zeros(1:nr,K_components,1:s_max)

			Gsrc = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)
			drGsrc = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)
			Gobs1 = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)

			# gℓω_r₁_rsrc = OffsetArray(zeros(ComplexF64,nr),1:nr)

			if r₁_ind != r₂_ind
				Gobs2 = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)
			end

			# temporary array to precompute the radial part, indices are (r,η)
			f_radial_0_r₁ = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1) # the η=-1 and η=1 terms are identical
			f_radial_1_r₁ = OffsetArray(zeros(ComplexF64,nr,3),1:nr,-1:1)
			fℓ′ℓsω_r₁ = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)

			if r₁_ind != r₂_ind
				f_radial_0_r₂ = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)
				f_radial_1_r₂ = OffsetArray(zeros(ComplexF64,nr,3),1:nr,-1:1)
				fℓ′ℓsω_r₂ = OffsetArray(zeros(ComplexF64,nr,2),1:nr,0:1)
			else
				fℓ′ℓsω_r₂ = view(fℓ′ℓsω_r₁,:,:)
			end

			# Clebsch Gordan coefficients, indices are (s,η,t)
			Cℓ′ℓ = OffsetArray(zeros(3,s_max,2),-1:1,1:s_max,-1:0)

			Dr = dbydr(dr)

			for proc_id in proc_id_range

				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")

				# Get a list of (ℓ,ω) in this file
				modes_in_src_Gfn_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_src_Gfn_file)

					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_src_Gfn_file,(ℓ,ω_ind))

					ω = 2π*ν_arr[ω_ind]
		    		
		    		G = read(Gsrc_file[1],:,:,1:2,1,ℓω_index_Gsrc_file)
		    		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		G_r₁_rsrc = Gsrc[r₁_ind,0]
		    		G_r₂_rsrc = Gsrc[r₂_ind,0]

		    		drGsrc[:,0] = Dr*Gsrc[:,0]
		    		drGsrc[:,1] = Dr*Gsrc[:,1]

				    for ℓ′ in abs(ℓ-s_max):ℓ+s_max

				    	if ℓ′ ∉ ℓ_arr
				    		continue
				    	end

				    	# ignore line-of-sight projection, just the β=γ=0 component
				    	Yℓ′ℓ_s0_n1n2 = BiPoSH_s0(ℓ′,ℓ,1:s_max,0,0,x1,x2)
			    		Yℓ′ℓ_s0_n2n1 = BiPoSH_s0(ℓ′,ℓ,1:s_max,0,0,x2,x1)

			    		# Compute the CG coefficients that appear in fℓ′ℓsω
			    		for t=-1:0,s in 1:s_max,η=-1:1
			    			Cℓ′ℓ[η,s,t] = clebschgordan(ℓ′,-η,ℓ,η+t,s,t)
			    		end

			    		proc_id_mode_Gobs1,ℓ′ω_index_Gobs1_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ′,ω_ind),num_procs_obs1)
			    		Gobs1_file = FITS(joinpath(Gfn_path_x1,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs1),"r")
						
	    				# Green functions based at the observation point for ℓ′
	    				G = read(Gobs1_file[1],:,:,1:2,1,ℓ′ω_index_Gobs1_file)
			    		@. Gobs1[:,0] = G[:,1,1] + im*G[:,2,1]
			    		@. Gobs1[:,1] = G[:,1,2] + im*G[:,2,2]

			    		close(Gobs1_file)

			    		if r₁_ind != r₂_ind

			    			proc_id_mode_Gobs2,ℓ′ω_index_Gobs2_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ′,ω_ind),num_procs_obs2)
			    			Gobs2_file = FITS(joinpath(Gfn_path_x2,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs2),"r")

				    		G = read(Gobs2_file[1],:,:,1:2,1,ℓ′ω_index_Gobs2_file)
				    		@. Gobs2[:,0] = G[:,1,1] + im*G[:,2,1]
				    		@. Gobs2[:,1] = G[:,1,2] + im*G[:,2,2]

				    		close(Gobs2_file)
				    	end

			    		G = nothing

			    		# precompute the radial term in f, α=0
			    		for η=0:1
			    			@. f_radial_0_r₁[:,η] = (-1)^η * Gobs1[:,η] * drGsrc[:,η]
			    			if r₁_ind != r₂_ind
			    				@. f_radial_0_r₂[:,η] = (-1)^η * Gobs2[:,η] * drGsrc[:,η]
			    			end
			    		end

			    		# α=1
			    		for η=-1:1
			    			@. f_radial_1_r₁[:,η] = (-1)^η * 1/r * Gobs1[:,abs(η)] * 
			    							( Ω(ℓ,η)*Gsrc[:,abs(η)] - ((η != -1) ? Gsrc[:,abs(η-1)] : 0) )
			    			if r₁_ind != r₂_ind
			    				@. f_radial_1_r₂[:,η] = (-1)^η * 1/r * Gobs2[:,abs(η)] * 
			    							( Ω(ℓ,η)*Gsrc[:,abs(η)] - ((η != -1) ? Gsrc[:,abs(η-1)] : 0) )
			    			end
			    		end

			    		for s in LinearIndices(Yℓ′ℓ_s0_n2n1)
			    			# radial component (for all s)
			    			
			    			fℓ′ℓsω_r₁[:,0] .= sum(f_radial_0_r₁[:,abs(η)]*Cℓ′ℓ[η,s,0] for η=-1:1) 
			    			fℓ′ℓsω_r₁[:,1] .= sum(f_radial_1_r₁[:,η]*Cℓ′ℓ[η,s,-1] for η=-1:1)

			    			if r₁_ind != r₂_ind
			    				fℓ′ℓsω_r₂[:,0] .= sum(f_radial_0_r₂[:,abs(η)]*Cℓ′ℓ[η,s,0] for η=-1:1) 
			    				fℓ′ℓsω_r₂[:,1] .= sum(f_radial_1_r₂[:,η]*Cℓ′ℓ[η,s,-1] for η=-1:1)
			    			end

			    			if isodd(ℓ+ℓ′+s) && -1 in K_components
			    				# tangential - component (K⁺ - K⁻), only for odd l+l′+s
								# only imag part calculated, the actual kernel is iK
								# extra factor of 2 from the (1 - (-1)^(ℓ+ℓ′+s)) term
								@. K[:,-1,s] += 2 * dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
					     					real(conj(h_ω_arr[ω_ind]) *
					     					(conj(fℓ′ℓsω_r₁[:,1])*G_r₂_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
					     						+ conj(G_r₁_rsrc)*fℓ′ℓsω_r₂[:,1]*Yℓ′ℓ_s0_n2n1[s]) )
			    			end

			    			if 0 in K_components
						     	@. K[:,0,s] +=  dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
						     					real(conj(h_ω_arr[ω_ind]) *
						     					(conj(fℓ′ℓsω_r₁[:,0])*G_r₂_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
						     						+ conj(G_r₁_rsrc)*fℓ′ℓsω_r₂[:,0]*Yℓ′ℓ_s0_n2n1[s]) )
						    end
							
							if iseven(ℓ+ℓ′+s) && 1 in K_components
								# tangential + component (K⁺ + K⁻), only for even l+l′+s
								# extra factor of 2 from the (1 + (-1)^(ℓ+ℓ′+s)) term
								@. K[:,1,s] +=  2 * dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
					     					real(conj(h_ω_arr[ω_ind]) *
					     					(conj(fℓ′ℓsω_r₁[:,1])*G_r₂_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
					     						+ conj(G_r₁_rsrc)*fℓ′ℓsω_r₂[:,1]*Yℓ′ℓ_s0_n2n1[s]) )
							end
					    end

					end

				end

				close(Gsrc_file)
				
			end

			return @. 4r^2 * ρ * K
		end

		procs = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs)
		# println("Submitting to workers $procs")
		# kernel_futures = [@spawnat p sum_modes(rank) for (rank,p) in enumerate(procs)]

		# axes.(fetch.(kernel_futures))
		# sum(fetch.(kernel_futures))

		Profile.clear()
		@spawn @profile sum_modes(1)
	end

	function flow_kernels_srange_t0(n1::Point2D,n2::Point2D,s_max;r_obs=Rsun-75e5,kwargs...)
		flow_kernels_srange_t0(Point3D(r_obs,n1),Point3D(r_obs,n2),s_max;kwargs...)
	end

	function meridional_flow_stream_function_kernel_srange(x1::SphericalPoint,x2::SphericalPoint,s_max;kwargs...)
		Kv = flow_kernels_srange_t0(x1,x2,s_max;K_components=0:1,kwargs...)
		Dr = dbydr(dr)

		Kψ_imag = zeros(nr,s_max)

		# negative of the imaginary part of the stream function kernel
		for s in 1:s_max
			Kψ_imag[:,s] .= Dr*(Kv[:,1,s] ./ρ) .+ Kv[:,1,s]./(ρ.*r) .- 2 .*Ω(s,0).*Kv[:,0,s]./(ρ.*r)
		end

		return (-im).*Kψ_imag
	end

end

module kernel3D
	using Main.kernel
	import WignerD: Ylmatrix

	function flow_kernels_rθ_slice(x1::SphericalPoint,x2::SphericalPoint,s_max;kwargs...)
		Kjlm_r = flow_kernels_srange(x1,x2,s_max;kwargs...)
		θ_arr = LinRange(0,π,nθ)
		# Slice at ϕ=0
		n_arr = [Point2D(θ,0) for θ in θ_arr]

		K2D_r_θ = zeros(size(Kjlm_r,1))

		# The 3D kernel is given by ∑ Ks0n(r) [Ys0n(θ,ϕ)]†
		# This has components [Ys0n(θ,ϕ)]* = (-1)^(n) Ys0n(θ,ϕ)


	end
end