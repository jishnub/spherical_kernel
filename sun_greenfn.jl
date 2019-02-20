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
		if num_procs == 1
			return arr₁
		end

		# assume proc_id should start from 1
		# for workers, subtract 1 from myid()

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
			return -1
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

	function get_processor_range_from_split_array(arr₁,arr₂,tasks_on_proc,num_procs)
		
		if isempty(tasks_on_proc)
			return 0:-1
		end

		tasks_arr = collect(tasks_on_proc)
		proc_id_start = get_processor_id_from_split_array(arr₁,arr₂,first(tasks_arr),num_procs)
		proc_id_end = get_processor_id_from_split_array(arr₁,arr₂,last(tasks_arr),num_procs)
		return proc_id_start:proc_id_end
	end

	function split_product_and_get_local_processor_range(arr₁,arr₂,num_procs_original,num_procs=nworkers(),proc_id=worker_rank())
		tasks = split_product_across_processors(arr₁,arr₂,num_procs,proc_id)
		procs = get_processor_range_from_split_array(arr₁,arr₂,tasks,num_procs_original)
	end

	workers_active(arr₁,arr₂) = [p for (rank,p) in enumerate(workers()) if !isempty(split_product_across_processors(arr₁,arr₂,nworkers(),rank))]

	export split_product_across_processors,get_processor_id_from_split_array,get_processor_range_from_split_array,workers_active
end

###########################################################################################

###########################################################################################

# Operators

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
	@reexport using Main.finite_difference,DelimitedFiles,Polynomials,Main.parallel_utilities

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
		c = modelS[flip,2];
		ρ = modelS[flip,3];

		G = 6.67428e-8 # cgs units
		m = Msun.*exp.(modelS_detailed[flip,2])

		g = @. G*m/r^2

		N2 = @. g * modelS_detailed[flip,15] / r

		return Rsun,nr,r,dr,c,ρ,g,N2,γ_damping
	end

	export load_solar_model
	const Rsun,nr,r,dr,c,ρ,g,N2,γ_damping = load_solar_model()
	export Rsun,nr,r,dr,c,ρ,g,N2,γ_damping
end


#################################################################
# Green function radial components, main function
#################################################################

module Greenfn_radial

	using Reexport
	@reexport using Main.load_parameters
	using Main.parallel_utilities
	using LinearAlgebra,SparseArrays,Main.finite_difference
	@reexport using OffsetArrays, JLD2,Printf

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

	struct Gfn
		mode :: NamedTuple{(:ω, :ω_ind, :ℓ),Tuple{Float64,Int64,Int64}}
		G :: OffsetArray{ComplexF64,3,Array{ComplexF64,3}}
	end

	function compute_radial_component_of_Gfn_onemode(ω,ℓ;r_src=Rsun-75e5)

		# Solar model

		Sr,Sh = source(ω,ℓ,r_src=r_src)

		M = ℒωℓr(ω,ℓ);

		H = M\Sr;

		αrℓω = H[1:nr-1]
		prepend!(αrℓω,0)

		βrℓω = H[nr:end]
		append!(βrℓω,0)

		H = M\Sh;

		αhℓω = H[1:nr-1]
		prepend!(αhℓω,0)

		βhℓω = H[nr:end]
		append!(βhℓω,0)

		return αrℓω,βrℓω,αhℓω,βhℓω
	end


	function Gfn_path_from_source_radius(r_src::Real)
		user=ENV["USER"]
		scratch=get(ENV,"SCRATCH","/scratch/$user")
		return "$scratch/Greenfn_src$((r_src/Rsun > 0.99 ? 
										(@sprintf "%dkm" (Rsun-r_src)/1e5) : (@sprintf "%.2fRsun" r_src/Rsun) ))"
	end

	function compute_Greenfn_radial_components_allmodes_parallel(r_src=Rsun-75e5;ℓ_arr=1:100,ν_low=2.0e-3,ν_high=4.5e-3)

		Nν = 3750; Nt = 2*(Nν-1)
		ν_full = LinRange(0,7.5e-3,Nν); ν_nyquist = ν_full[end];
		ν_arr = ν_full[ν_low .≤ ν_full .≤ ν_high];
		dν = ν_full[2] - ν_full[1]; T=1/dν; dt = T/Nt;
		ν_start_zeros = count(ν_full .<= ν_low)
		ν_end_zeros = count(ν_high .<= ν_full)
		println("$(length(ν_arr)) frequencies over $(@sprintf "%.1e" ν_arr[1]) to $(@sprintf "%.1e" ν_arr[end])")

		ω_arr = 2π .* ν_arr;

		println("ℓ from $(ℓ_arr[1]) to $(ℓ_arr[end])")

		Gfn_save_directory = Gfn_path_from_source_radius(r_src)

		if !isdir(Gfn_save_directory)
			mkdir(Gfn_save_directory)
		end
		println("Saving output to $Gfn_save_directory")

		num_procs = nworkers()
		println("Number of workers: $num_procs")

		@save joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs ν_full dν ν_start_zeros ν_end_zeros Nν Nt dt ν_nyquist T

		function compute_G_somemodes_serial_oneproc(rank)

			# Each processor will save all ℓ's  for a range of frequencies if number of frequencies can be split evenly across processors.
			ℓ_ω_proc = split_product_across_processors(ℓ_arr,axes(ν_arr,1),nworkers(),rank);
			
			Gfn_arr = Vector{Gfn}(undef,length(ℓ_ω_proc));

			G = OffsetArray(zeros(ComplexF64,nr,2,2),1:nr,0:1,0:1)
			Dr = dbydr(dr) # derivative operator
			T2 = (N2./g .+ Dr*log.(r.^2 .*ρ) )

			for (ind,(ℓ,ω_ind)) in enumerate(ℓ_ω_proc)
				
				ω = ω_arr[ω_ind]
				αr,βr,αh,βh = compute_radial_component_of_Gfn_onemode(ω,ℓ);

				@. G[:,0,0] = αr
				@. G[:,1,0] = Ω(ℓ,0) * βr/(ρ*r*ω^2)

				@. G[:,0,1] = αh / √2
				
				G[:,1,1] .=  r./√(ℓ*(ℓ+1)) .* (βh./(ρ.*c.^2) .+ T2.*αh + Dr*αh) ./2
				
				Gfn_arr[ind] = Gfn((ω=ω,ω_ind=ω_ind,ℓ=ℓ),copy(G));
			end

			save_path = joinpath(Gfn_save_directory,@sprintf "Gfn_proc_%03d.jld2" rank)
			@save save_path Gfn_arr
			# return Gfn_arr # useful for debugging
		end

		futures = [@spawnat p compute_G_somemodes_serial_oneproc(rank) for (rank,p) in enumerate(workers_active(ℓ_arr,ν_arr))]

		@time fetch.(futures);
	end

	export compute_Greenfn_radial_components_allmodes_parallel,Gfn,Gfn_path_from_source_radius,Ω
end

#################################################################
# Full 3D Green function
#################################################################

module crosscov
	
	using Reexport,Distributed
	using Interpolations,FFTW,JLD2,Printf,OffsetArrays,PyCall
	using FastGaussQuadrature,DSP,LsqFit
	using Main.parallel_utilities
	import Main.Greenfn_radial: Gfn_path_from_source_radius,Gfn
	using Main.finite_difference
	@pyimport scipy.integrate as integrate
	@reexport using Legendre
	@reexport using PointsOnASphere
	import PyPlot; plt=PyPlot

	include("./twopoint_functions_on_a_sphere.jl");
	@reexport using .Sphere_2point_functions
	include("./vector_fields_on_a_sphere.jl")
	@reexport using .Sphere_vectorfields

	export Cω,Cϕω,Cω_onefreq,h,Powspec,Ct
	export δCω_uniform_rotation_firstborn_integrated_over_angle
	export δCω_uniform_rotation_rotatedwaves_linearapprox
	export δCω_uniform_rotation_rotatedwaves,δCt_uniform_rotation_rotatedwaves

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

	    compute_3D_Greenfn_helicity_angular_sections_onefreq(x′,Gfn_radial_arr,θ_ϕ_iterator,ℓmax,ν,procid)
    end

    function compute_3D_Greenfn_helicity_angular_sections_onefreq(x′,Gfn_radial_arr::Vector{Gfn},θ_ϕ_iterator,ℓmax::Integer,ν::Real=3e-3,procid=myid()-1)

		Gfn_save_directory = Gfn_path_from_source_radius(x′)

    	Gfn_3D_arr = OffsetArray{ComplexF64}(undef,1:nr,1:length(θ_ϕ_iterator),-1:1)
    	fill!(Gfn_3D_arr,0)

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

    function compute_3D_Greenfn_spherical_angular_sections_onefreq(x′,
    	θ_full::AbstractArray,ϕ_full::AbstractArray,ν::Real=3e-3,procid=myid()-1)

		Gfn_save_directory = Gfn_path_from_source_radius(x′)

    	@load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr ℓ_arr num_procs
    	
    	Gfn_radial_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_save_directory,ν,ν_arr,ℓ_arr,num_procs)

    	ℓmax = ℓ_arr[end]

    	θ_ϕ_iterator = split_product_across_processors(θ_full,ϕ_full,nworkers(),procid)

	    compute_3D_Greenfn_spherical_angular_sections_onefreq(x′,Gfn_radial_arr,θ_ϕ_iterator,ℓmax,ν,procid) 
    end

    function compute_3D_Greenfn_spherical_angular_sections_onefreq(x′,Gfn_radial_arr::Vector{Gfn},θ_ϕ_iterator,ℓmax::Integer,ν::Real=3e-3,procid=myid()-1)

		Gfn_save_directory = Gfn_path_from_source_radius(x′)

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

    function δC_uniform_rotation_spherical_angular_sections_onefreq_without_metric_factors(x1,x2,θ_full,ϕ_full,ν=3e-3,procid=myid()-1)

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

				(ℓ < 2) && continue

				G00 = view(Gfn.G,:,1)
				G10 = view(Gfn.G,:,2)

				@. integrand[1:nr,θϕ_ind] += (2ℓ+1)/4π * (

				- G00[r₂_ind]*(
					conj(Gfn3D_x1[1:nr,θϕ_ind,1])*conj(G00) * ∂ϕPl2[ℓ]  + 
					conj(Gfn3D_x1[1:nr,θϕ_ind,2])*conj(G10)/Ω(ℓ,0) * ∂ϕ∂θPl2[ℓ] +
					conj(Gfn3D_x1[1:nr,θϕ_ind,3])*conj(G10)/Ω(ℓ,0)* ∇ϕ∂ϕPl2[ℓ]
					) +

				conj(G00[r₁_ind])*(
					Gfn3D_x2[1:nr,θϕ_ind,1]*G00 * ∂ϕPl1[ℓ]  + 
					Gfn3D_x2[1:nr,θϕ_ind,2]*G10/Ω(ℓ,0) * ∂ϕ∂θPl1[ℓ] +
					Gfn3D_x2[1:nr,θϕ_ind,3]*G10/Ω(ℓ,0) * ∇ϕ∂ϕPl1[ℓ] 
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

		tasks_on_proc = split_product_across_processors(ℓ_arr,ν_ind_range)

		proc_id_range = get_processor_range_from_split_array(ℓ_arr,axes(ν_arr,1),tasks_on_proc,num_procs);
		

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

	#######################################################################################################

	function Cω(x1::Point3D,x2::Point3D;ℓ_range=nothing,ν_ind_range=nothing)
		
		Gfn_path = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs ν_full ν_start_zeros

		Nν_full = length(ν_full)
		Cω_arr = zeros(ComplexF64,Nν_full)
		Nν = length(ν_arr)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = max(ℓ_arr[1],ℓ_range[1]):min(ℓ_arr[end],ℓ_range[end])
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν
		else
			ν_ind_range = max(1,ν_ind_range[1]):min(Nν,ν_ind_range[end])
		end

		function Cω_summodes(rank)

			tasks_on_proc = Main.parallel_utilities.split_product_across_processors(ℓ_range,ν_ind_range,nworkers(),rank) |> collect
			proc_range = Main.parallel_utilities.get_processor_range_from_split_array(ℓ_arr,1:Nν,tasks_on_proc,num_procs)

			if isempty(tasks_on_proc)
				return OffsetArray(zeros(ComplexF64,1),0)
			end

			ν_ind_min = first(tasks_on_proc)[2]
			ν_ind_max = last(tasks_on_proc)[2]
			Nν_on_proc = ν_ind_max - ν_ind_min + 1

			Cω_proc = OffsetArray(zeros(ComplexF64,Nν_on_proc),ν_ind_min:ν_ind_max)

			Pl_cosχ = Pl(cosχ(x1,x2),ℓmax=ℓ_arr[end])

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind = argmin(abs.(r .- x2.r))

			for proc_id in proc_range
				G_proc_file = joinpath(Gfn_path,@sprintf "Gfn_proc_%03d.jld2" proc_id)
				@load G_proc_file Gfn_arr
		    	for Gfn in Gfn_arr
		    		ℓ = Gfn.mode.ℓ
		    		(ℓ==0) && continue
		    		ω = Gfn.mode.ω
		    		ω_ind = Gfn.mode.ω_ind
		    		
		    		((ℓ,ω_ind) ∉ tasks_on_proc) && continue

				    α = Gfn.G[:,0,0]
				    
				    Cω_proc[ω_ind] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α[r₁_ind]) * α[r₂_ind] * Pl_cosχ[ℓ]
				end
			end

			return Cω_proc
		end

		futures = [@spawnat p Cω_summodes(rank) for (rank,p) in enumerate(workers_active(ℓ_range,ν_ind_range))]

		for f in futures
			Ci = fetch(f)
			ax = axes(Ci,1)
			@. Cω_arr[ax + ν_start_zeros] += Ci.parent
		end

		# G_proc_file = joinpath(Gfn_path,@sprintf "Gfn_proc_%03d.jld2" 1)
		# @load G_proc_file Gfn_arr

		return Cω_arr
	end

	Cω(n1::Point2D,n2::Point2D;r_obs::Real=Rsun-75e5,ℓ_range=nothing,ν_ind_range=nothing) = Cω(Point3D(r_obs,n1),Point3D(r_obs,n2),ℓ_range=nothing,ν_ind_range=nothing)
	Cω(Δϕ::Real;r_obs::Real=Rsun-75e5,ℓ_range=nothing,ν_ind_range=nothing) = Cω(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ),ℓ_range=nothing,ν_ind_range=nothing)

	function ∂ϕ₂Cω(x1::Point3D,x2::Point3D;ℓ_range=nothing,ν_ind_range=nothing)
		
		Gfn_path = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs ν_full ν_start_zeros

		Nν = length(ν_arr)
		Nν_full = length(ν_full)

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = max(ℓ_arr[1],ℓ_range[1]):min(ℓ_arr[end],ℓ_range[end])
		end

		if isnothing(ν_ind_range)
			ν_ind_range = 1:Nν
		end

		function Cω_summodes(rank)

			tasks_on_proc = Main.parallel_utilities.split_product_across_processors(ℓ_range,ν_ind_range,nworkers(),rank) |> collect
			proc_range = Main.parallel_utilities.get_processor_range_from_split_array(ℓ_arr,1:Nν,tasks_on_proc,num_procs)

			if isempty(tasks_on_proc)
				return OffsetArray(zeros(ComplexF64,1),0)
			end

			ν_ind_min = first(tasks_on_proc)[2]
			ν_ind_max = last(tasks_on_proc)[2]
			Nν_on_proc = ν_ind_max - ν_ind_min + 1

			Cω_proc = OffsetArray(zeros(ComplexF64,Nν_on_proc),ν_ind_min:ν_ind_max)

			∂ϕ₂Pl_cosχ = dPl(cosχ(x1,x2),ℓmax=ℓ_arr[end]) .* ∂ϕ₂cosχ(x1,x2)

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind = argmin(abs.(r .- x2.r))

			for proc_id in proc_range
				G_proc_file = joinpath(Gfn_path,@sprintf "Gfn_proc_%03d.jld2" proc_id)
				@load G_proc_file Gfn_arr
		    	for Gfn in Gfn_arr
		    		ℓ = Gfn.mode.ℓ
		    		(ℓ==0) && continue
		    		ω = Gfn.mode.ω
		    		ω_ind = Gfn.mode.ω_ind
		    		
		    		((ℓ,ω_ind) ∉ tasks_on_proc) && continue

				    α = Gfn.G[:,0,0]
				    
				    Cω_proc[ω_ind] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α[r₁_ind]) * α[r₂_ind] * ∂ϕ₂Pl_cosχ[ℓ]
				end
			end

			return Cω_proc
		end

		futures = [@spawnat p Cω_summodes(rank) for (rank,p) in enumerate(workers_active(ℓ_range,ν_ind_range))]

		Cω_arr = zeros(ComplexF64,Nν_full)

		for f in futures
			Ci = fetch(f)
			ax = axes(Ci,1)
			@. Cω_arr[ax + ν_start_zeros] += Ci.parent
		end

		return Cω_arr
	end

	∂ϕ₂Cω(n1::Point2D,n2::Point2D;r_obs=Rsun-75e5,ℓ_range=nothing,ν_ind_range=nothing) = ∂ϕ₂Cω(Point3D(r_obs,n1),Point3D(r_obs,n2),ℓ_range=nothing,ν_ind_range=nothing)
	∂ϕ₂Cω(Δϕ::Real;r_obs=Rsun-75e5,ℓ_range=nothing,ν_ind_range=nothing) = ∂ϕ₂Cω(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ),ℓ_range=nothing,ν_ind_range=nothing)

	function Ct(x1::Point3D,x2::Point3D;ℓ_range=nothing,dν=nothing)
		if isnothing(dν)
			Gfn_path = Gfn_path_from_source_radius(x1)
			@load "$Gfn_path/parameters.jld2" dν
		end
		C = Cω(x1,x2,ℓ_range=ℓ_range)
		Nt = 2*(length(C)-1)
		brfft(C,Nt).*dν
	end

	Ct(n1::Point2D,n2::Point2D;r_obs=Rsun-75e5,ℓ_range=nothing,dν=nothing) = Ct(Point3D(r_obs,n1),Point3D(r_obs,n2),ℓ_range=ℓ_range,dν=dν)
	Ct(Δϕ::Real;r_obs=Rsun-75e5,ℓ_range=nothing,dν=nothing) = Ct(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ),ℓ_range=ℓ_range,dν=dν)

	function ∂ϕ₂Ct(x1::Point3D,x2::Point3D;ℓ_range=nothing,dν=nothing)
		if isnothing(dν)
			Gfn_path = Gfn_path_from_source_radius(x1)
			@load "$Gfn_path/parameters.jld2" dν
		end
		C = ∂ϕ₂Cω(x1,x2,ℓ_range=ℓ_range)
		Nt = 2*(length(C)-1)
		brfft(C,Nt).*dν
	end

	∂ϕ₂Ct(n1::Point2D,n2::Point2D;r_obs=Rsun-75e5,ℓ_range=nothing,dν=nothing) = ∂ϕ₂Ct(Point3D(r_obs,n1),Point3D(r_obs,n2),ℓ_range=ℓ_range,dν=dν)
	∂ϕ₂Ct(Δϕ::Real;r_obs=Rsun-75e5,ℓ_range=nothing,dν=nothing) = ∂ϕ₂Ct(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ),ℓ_range=ℓ_range,dν=dν)

	function CΔϕω(r₁::Real=Rsun-75e5,r₂::Real=Rsun-75e5;ℓ_range=nothing)

		Gfn_path = Greenfn_radial.Gfn_path_from_source_radius(r₁)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs ν_full ν_start_zeros

		Nν = length(ν_arr)

		r₁_ind = argmin(abs.(r .- r₁))
		r₂_ind = argmin(abs.(r .- r₂))

		if !isnothing(ℓ_range)
			ℓ_range = max(ℓ_range[1],ℓ_arr[1]):min(ℓ_range[end],ℓ_arr[end])
		else
			ℓ_range = ℓ_arr
		end

		ℓmax = ℓ_range[end]
		nϕ = 2*ℓmax
		Δϕ_arr = LinRange(0,2π,nϕ+1)[1:end-1]

		Cϕω_arr = zeros(ComplexF64,nϕ,length(ν_full))

		Pl_cosχ = OffsetArray(zeros(ℓmax+1,nϕ),(-1,0))
		
		for (ϕ_ind,Δϕ) in enumerate(Δϕ_arr)
			Pl_cosχ[:,ϕ_ind] = Pl(cos(Δϕ),ℓmax=ℓmax)
		end

		Pl_cosχ = collect(transpose(Pl_cosχ))

		function Cϕω_summodes(rank)
			tasks_on_proc = Main.parallel_utilities.split_product_across_processors(ℓ_range,1:Nν,nworkers(),rank)
			proc_range = Main.parallel_utilities.get_processor_range_from_split_array(ℓ_arr,1:Nν,tasks_on_proc,num_procs)

			ν_min_ind = first(collect(tasks_on_proc))[2]
			ν_max_ind = last(collect(tasks_on_proc))[2]
			Nν_on_proc = ν_max_ind - ν_min_ind + 1

			Cϕω_arr = OffsetArray(zeros(ComplexF64,nϕ,Nν_on_proc),(1:nϕ,ν_min_ind:ν_max_ind))

			for proc_id in proc_range
				G_proc_file = joinpath(Gfn_path,@sprintf "Gfn_proc_%03d.jld2" proc_id)
		    	@load G_proc_file Gfn_arr
		    	for Gfn in Gfn_arr
		    		ℓ = Gfn.mode.ℓ
		    		ω = Gfn.mode.ω
		    		ω_ind = Gfn.mode.ω_ind
		    		
		    		((ℓ==0) || ((ℓ,ω_ind) ∉ tasks_on_proc) ) && continue

				    α = view(Gfn.G,:,1)
				    
				    for ϕ_ind in 1:nϕ
				    	Cϕω_arr[ϕ_ind,ω_ind] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α[r₁_ind]) * α[r₂_ind] * Pl_cosχ[ϕ_ind,ℓ]
				    end
				end
			end

			return Cϕω_arr
		end

	    futures = [@spawnat p Cϕω_summodes(rank) for (rank,p) in enumerate(workers_active(ℓ_range,1:Nν))]

	    for f in futures
	    	Ci = fetch(f)
	    	ν_inds = axes(Ci,2)
	    	@. Cϕω_arr[:,ν_inds .+ ν_start_zeros] += Ci.parent
	    end

		return Cϕω_arr
	end

	function CΔϕt(r₁::Real=Rsun-75e5,r₂::Real=Rsun-75e5;ℓ_range=nothing,dν=nothing) 
		C = CΔϕω(r₁,r₂,ℓ_range=ℓ_range)
		if isnothing(dν)
			Gfn_path = Gfn_path_from_source_radius(r₁)
			@load "$Gfn_path/parameters.jld2" dν
		end
		Nω = size(C,2)
		Nt = 2*(Nω-1)
		return brfft(C,Nt,2)*dν
	end

	Cmω(r₁::Real=Rsun-75e5,r₂::Real=Rsun-75e5;ℓ_range=nothing) = fft(CΔϕω(r₁,r₂,ℓ_range=ℓ_range),1)

	function Cω_onefreq(x1::Point3D,x2::Point3D,ν=3e-3)
		
		Gfn_path = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs

		ν_test_ind = argmin(abs.(ν_arr .- ν))
		Nν = length(ν_arr)

		@printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_arr[ν_test_ind] ℓ_arr[1] ℓ_arr[end]

		Cν = zero(ComplexF64)

		r₁_ind = argmin(abs.(r .- x1.r))
		r₂_ind = argmin(abs.(r .- x2.r))

		Pl_cosχ = Pl(cosχ(x1,x2),ℓmax=ℓ_arr[end])

		function Cω_summodes(rank)
			
			tasks_on_proc = Main.parallel_utilities.split_product_across_processors(ℓ_arr,ν_test_ind:ν_test_ind,nworkers(),rank)
			proc_id_start,proc_id_end = Main.parallel_utilities.get_processor_range_from_split_array(ℓ_arr,1:Nν,tasks_on_proc,num_procs)

			Cν = zero(ComplexF64)

			for proc_id in proc_id_start:proc_id_end
				G_proc_file = joinpath(Gfn_path,@sprintf "Gfn_proc_%03d.jld2" proc_id)
				@load G_proc_file Gfn_arr

				for Gfn in Gfn_arr
					ℓ = Gfn.mode.ℓ
					ω = Gfn.mode.ω

				    α = view(Gfn.G,:,1)
				    (ℓ==0) && continue
				    Cν += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α[r₁_ind]) * α[r₂_ind] * Pl_cosχ[ℓ]
				end

			end

			return Cν

		end

		Cν = @distributed (+) for rank in 1:nworkers()
				Cω_summodes(rank)
			end

		return Cν
	end

	function Cω_onefreq_ℓspectrum(x1::Point3D,x2::Point3D,ν=3e-3;ℓ_range=nothing)
		
		Gfn_path = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs

		if !isnothing(ℓ_range)
			ℓ_range = max(ℓ_range[1],ℓ_arr[1]):min(ℓ_range[end],ℓ_arr[end])
		else
			ℓ_range = ℓ_arr
		end

		ν_test_ind = argmin(abs.(ν_arr .- ν))
		Nν = length(ν_arr)

		# @printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_arr[ν_test_ind] ℓ_range[1] ℓ_range[end]

		# Cν_ℓ = OffsetArray(zeros(ComplexF64,length(ℓ_range)),ℓ_range)

		r₁_ind = argmin(abs.(r .- x1.r))
		r₂_ind = argmin(abs.(r .- x2.r))

		Pl_cosχ = Pl(cosχ(x1,x2),ℓmax=ℓ_range[end])

		# Gfn_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_path,ν,ν_arr,ℓ_arr,num_procs)

		function Cω_summodes(rank)
			
			tasks_on_proc = Main.parallel_utilities.split_product_across_processors(ℓ_range,ν_test_ind:ν_test_ind,nworkers(),rank)
			proc_range = Main.parallel_utilities.get_processor_range_from_split_array(ℓ_arr,1:Nν,tasks_on_proc,num_procs)

			Cν_ℓ = OffsetArray(zeros(ComplexF64,length(ℓ_range)),ℓ_range)

			for proc_id in proc_range
				G_proc_file = joinpath(Gfn_path,@sprintf "Gfn_proc_%03d.jld2" proc_id)
				@load G_proc_file Gfn_arr

				for Gfn in Gfn_arr
					ℓ = Gfn.mode.ℓ
					ω = Gfn.mode.ω
					ω_ind = Gfn.mode.ω_ind

				    α = view(Gfn.G,:,1)
				    (ℓ==0) || ((ℓ,ω_ind) ∉ tasks_on_proc) && continue

				    Cν_ℓ[ℓ] = ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α[r₁_ind]) * α[r₂_ind] * Pl_cosχ[ℓ]
				end

			end

			return Cν_ℓ

		end

		Cν_ℓ = sum(Cω_summodes(rank) for (rank,p) in enumerate(workers_active(ℓ_range,ν_test_ind:ν_test_ind)) ) 

	end

	Cω_onefreq_ℓspectrum(Δϕ::Real=π/3,ν=3e-3;ℓ_range=nothing,r_obs=Rsun-75e5) = Cω_onefreq_ℓspectrum(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ),ℓ_range=ℓ_range)
	

	function Cτ_rotating(x1::Point3D,x2::Point3D;Ω_rot = 20e2/Rsun,τ_ind_arr = nothing)
		
		# Return C(Δϕ,ω) = RFFT(C(Δϕ,τ)) = RFFT(IRFFT(C0(Δϕ - Ωτ,ω))(τ))

		Gfn_path = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path,"parameters.jld2") ν_full dν

		nω = length(ν_full)
		Nt = 2*(nω-1)
		T = 1/dν
		dt = T/Nt

		if isnothing(τ_ind_arr)
			τ_ind_arr = 1:div(Nt,2)
		end

		Nτ = length(τ_ind_arr)

		Cτ_arr = OffsetArray(zeros(Nτ),τ_ind_arr)
	    
		for τ_ind in τ_ind_arr
			τ = (τ_ind-1) * dt
	    	x2′ = Point3D(x2.r,x2.θ,x2.ϕ-Ω_rot*τ)
	    	Cτ_arr[τ_ind] = Ct(x1,x2′,dν=dν)[τ_ind]
	    end

		return Cτ_arr
	end

	Cτ_rotating(n1::Point2D,n2::Point2D;r_obs=Rsun-75e5,Ω_rot = 20e2/Rsun,τ_ind_arr = nothing) = Cτ_rotating(Point3D(r_obs,n1),Point3D(r_obs,n2),Ω_rot = Ω_rot,τ_ind_arr = τ_ind_arr)
	Cτ_rotating(Δϕ::Real;r_obs=Rsun-75e5,Ω_rot = 20e2/Rsun,τ_ind_arr = nothing) = Cτ_rotating(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ),Ω_rot = Ω_rot,τ_ind_arr = τ_ind_arr)

	
	# function δCω_uniform_rotation_firstborn(x1 = Point3D(Rsun-75e5,π/2,0),x2=Point3D(Rsun-75e5,π/2,π/3),ν=3e-3;Ω_rot = 20e2/Rsun)
		
	# 	Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

	# 	@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs

	# 	ν_on_grid = ν_arr[argmin(abs.(ν_arr .- ν))]
	# 	ω = 2π*ν_on_grid

	# 	ℓmax = ℓ_arr[end]
	# 	nθ,nϕ = 2ℓmax,2ℓmax

	# 	θ = LinRange(0,π,nθ)
	# 	ϕ = LinRange(0,2π,nϕ)

	# 	workers_active = [i for i in workers() if length(split_product_across_processors(1:nθ,1:nϕ,nworkers(),i-1))!=0]

	# 	dC_Ω_futures = [@spawnat p δC_uniform_rotation_helicity_angular_sections_onefreq(x1,x2,nθ,nϕ,ν) for p in workers_active]

	# 	dC_Ω = reshape(vcat(fetch.(dC_Ω_futures)...),nθ,nϕ)

	# 	dC = -√2 *Ω_rot* ω^3 * Powspec(ω) * ∮dΩ(dC_Ω,θ,ϕ)
	# end

	# function δCω_uniform_rotation_firstborn_krishnendu(x1 = Point3D(Rsun-75e5,π/2,0),x2=Point3D(Rsun-75e5,π/2,π/3),ν=3e-3;Ω_rot = 20e2/Rsun)
	# 	Gfn_directory_x1 = Gfn_path_from_source_radius(x1)

	# 	@load joinpath(Gfn_directory_x1,"parameters.jld2") ν_arr ℓ_arr num_procs

	# 	ν_on_grid = ν_arr[argmin(abs.(ν_arr .- ν))]
	# 	ω = 2π*ν_on_grid

	# 	@printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_arr[1] ℓ_arr[end]

	# 	ℓmax = ℓ_arr[end]

	# 	θ = LinRange(0,π,4ℓmax)[2:end-1]; nθ = length(θ)
	# 	cosθGL,wGL = gausslegendre(4ℓmax); θGL=acos.(cosθGL); nθGL = length(θGL);
	# 	ϕ = LinRange(0,2π,4ℓmax); dϕ = ϕ[2] - ϕ[1]; nϕ=length(ϕ)

	# 	# linear θ grid
	# 	workers_active = [i for i in workers() if length(split_product_across_processors(1:nθ,1:nϕ,nworkers(),i-1))!=0]
	# 	dC_Ω_futures = [@spawnat p δC_uniform_rotation_spherical_angular_sections_onefreq(x1,x2,θ,ϕ,ν) for p in workers_active]
	# 	dC_Ω = reshape(vcat(fetch.(dC_Ω_futures)...),nθ,nϕ)

	# 	dC = 2im *Ω_rot* ω^3 * Powspec(ω) * integrate.simps(integrate.simps(dC_Ω.*sin.(θ),x=θ,axis=0),dx=dϕ)
	# 	println("Linear θ grid: δC = $dC")

	# 	# gauss-legendre nodes
	# 	workers_active = [i for i in workers() if length(split_product_across_processors(1:nθGL,1:nϕ,nworkers(),i-1))!=0]
	# 	dC_Ω_futures = [@spawnat p δC_uniform_rotation_spherical_angular_sections_onefreq(x1,x2,θGL,ϕ,ν) for p in workers_active]
	# 	dC_Ω_GL = reshape(vcat(fetch.(dC_Ω_futures)...),nθGL,nϕ)

	# 	dC = 2im *Ω_rot* ω^3 * Powspec(ω) * integrate.simps(dropdims(sum(wGL.*dC_Ω_GL,dims=1),dims=1),dx=dϕ)

	# 	println("Gauss-Legendre quadrature: δC = $dC")
	# end

	#######################################################################################################################################

	function δCω_uniform_rotation_firstborn_integrated_over_angle(x1::Point3D,x2::Point3D,ν::Real;Ω_rot = 20e2/Rsun,ℓ_range=nothing)
		Gfn_path = Gfn_path_from_source_radius(x1)
		r₁_ind = argmin(abs.(r .- x1.r))

		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs

		if !isnothing(ℓ_range)
			ℓ_range = max(ℓ_range[1],ℓ_arr[1]):min(ℓ_range[end],ℓ_arr[end])
		else
			ℓ_range = ℓ_arr
		end

		Nν = length(ν_arr)

		ν_test_ind = argmin(abs.(ν_arr .- ν))
		ν_on_grid = ν_arr[ν_test_ind]

		ω = 2π*ν_on_grid

		# @printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_range[1] ℓ_range[end]

		ℓmax = ℓ_arr[end]

		dPl_cosχ = dPl(cosχ(x1,x2),ℓmax=ℓmax)

		# Gfn_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_path,ν,ν_arr,ℓ_arr,num_procs)

		δC = zero(ComplexF64)

		function δC_summodes(rank)
			tasks_on_proc = Main.parallel_utilities.split_product_across_processors(ℓ_range,ν_test_ind:ν_test_ind,nworkers(),rank)
			proc_id_range = Main.parallel_utilities.get_processor_range_from_split_array(ℓ_arr,1:Nν,tasks_on_proc,num_procs)

			if isempty(proc_id_range)
				return zero(ComplexF64)
			end

			δC_r = zeros(ComplexF64,nr)

			for proc_id in proc_id_range

				G_proc_file = joinpath(Gfn_path,@sprintf "Gfn_proc_%03d.jld2" proc_id)
				if !isfile(G_proc_file)
					continue
				end

				@load G_proc_file Gfn_arr

				for Gfn in Gfn_arr
					ℓ = Gfn.mode.ℓ
					ω_ind = Gfn.mode.ω_ind

					((ℓ==0) || ((ℓ,ω_ind) ∉ tasks_on_proc) ) && continue

				    G0 = view(Gfn.G,:,0,0)
				    G1 = view(Gfn.G,:,1,0)

				    δC_r .+= (2ℓ+1)/4π*dPl_cosχ[ℓ]*∂ϕ₂cosχ(x1,x2) .* (
				    			@. real(G0[r₁_ind] * conj( G0^2 - 2G0*G1/Ω(ℓ,0) + (ℓ*(ℓ+1)-1)*(G1/Ω(ℓ,0))^2 ) ) )
				end

			end

			return integrate.simps((@. r^2 * ρ * δC_r),x=r)

		end

		futures = [@spawnat p δC_summodes(rank) for (rank,p) in enumerate(workers_active(ℓ_range,ν_test_ind:ν_test_ind))]

		δC = sum(fetch.(futures))

		return -4im*Ω_rot*ω^3*Powspec(ω)*δC
	end

	function δCω_uniform_rotation_firstborn_integrated_over_angle(Δϕ::Real,ν::Real;Ω_rot = 20e2/Rsun,r_obs = Rsun - 75e5,ℓ_range=nothing)
		δCω_uniform_rotation_firstborn_integrated_over_angle(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ),ν,Ω_rot=Ω_rot,ℓ_range=ℓ_range)
	end

	function δCω_uniform_rotation_firstborn_integrated_over_angle(n1::Point2D,n2::Point2D,ν::Real;Ω_rot = 20e2/Rsun,r_obs = Rsun - 75e5,ℓ_range=nothing)
		δCω_uniform_rotation_firstborn_integrated_over_angle(Point3D(r_obs,n1),Point3D(r_obs,n2),ν,Ω_rot=Ω_rot,ℓ_range=ℓ_range)
	end

	function δCω_uniform_rotation_firstborn_integrated_over_angle(x1::Point3D,x2::Point3D;Ω_rot = 20e2/Rsun,ℓ_range=nothing)
		Gfn_path = Gfn_path_from_source_radius(x1)
		r₁_ind = argmin(abs.(r .- x1.r))

		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs

		if !isnothing(ℓ_range)
			ℓ_range = max(ℓ_range[1],ℓ_arr[1]):min(ℓ_range[end],ℓ_arr[end])
		else
			ℓ_range = ℓ_arr
		end

		Nν = length(ν_arr)

		# @printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_range[1] ℓ_range[end]

		ℓmax = ℓ_arr[end]

		∂ϕ₂Pl_cosχ = dPl(cosχ(x1,x2),ℓmax=ℓmax) .*∂ϕ₂cosχ(x1,x2)

		# Gfn_arr = load_Greenfn_radial_coordinates_onefreq(Gfn_path,ν,ν_arr,ℓ_arr,num_procs)

		δC = zeros(ComplexF64,Nν)

		function δC_summodes(rank)
			tasks_on_proc = collect(Main.parallel_utilities.split_product_across_processors(ℓ_range,1:Nν,nworkers(),rank))

			if isempty(tasks_on_proc)
				return OffsetArray(zeros(ComplexF64,1),1:1)
			end

			proc_range = Main.parallel_utilities.get_processor_range_from_split_array(ℓ_arr,1:Nν,tasks_on_proc,num_procs)

			ν_min_ind = first(collect(tasks_on_proc))[2]
			ν_max_ind = last(collect(tasks_on_proc))[2]
			Nν_on_proc = ν_max_ind - ν_min_ind + 1

			δC_r = OffsetArray(zeros(ComplexF64,nr,Nν_on_proc),1:nr,ν_min_ind:ν_max_ind)

			for proc_id in proc_range

				G_proc_file = joinpath(Gfn_path,@sprintf "Gfn_proc_%03d.jld2" proc_id)
				@load G_proc_file Gfn_arr

				for Gfn in Gfn_arr
					ℓ = Gfn.mode.ℓ
					ω_ind = Gfn.mode.ω_ind
					ω = Gfn.mode.ω

					((ℓ==0) || ((ℓ,ω_ind) ∉ tasks_on_proc) ) && continue

				    G0 = view(Gfn.G,:,0,0)
				    G1 = view(Gfn.G,:,1,0)

				    @. δC_r[:,ω_ind] .+= ω^3*Powspec(ω)* (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] .* (
				    			real(G0[r₁_ind] * conj( G0^2 - 2G0*G1/Ω(ℓ,0) + (ℓ*(ℓ+1)-1)*(G1/Ω(ℓ,0))^2 ) ) )
				end

			end

			return OffsetArray(integrate.simps((@. r^2 * ρ * δC_r.parent),x=r,axis=0),ν_min_ind:ν_max_ind)

		end

		futures = [@spawnat p δC_summodes(rank) for (rank,p) in enumerate(workers_active(ℓ_range,1:Nν))]

		for f in futures
			δC_i = fetch(f)
			ax = axes(δC_i,1)
			@. δC[ax] += δC_i
		end

		return @. -4im*Ω_rot*δC
	end

	function δCω_uniform_rotation_firstborn_integrated_over_angle(Δϕ::Real;Ω_rot = 20e2/Rsun,r_obs = Rsun - 75e5,ℓ_range=nothing)
		δCω_uniform_rotation_firstborn_integrated_over_angle(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ),Ω_rot=Ω_rot,ℓ_range=ℓ_range)
	end

	function δCω_uniform_rotation_firstborn_integrated_over_angle(n1::Point2D,n2::Point2D;Ω_rot = 20e2/Rsun,r_obs = Rsun - 75e5,ℓ_range=nothing)
		δCω_uniform_rotation_firstborn_integrated_over_angle(Point3D(r_obs,n1),Point3D(r_obs,n2),Ω_rot=Ω_rot,ℓ_range=ℓ_range)
	end

	########################################################################################################################################


	function δCω_uniform_rotation_rotatedwaves_linearapprox(x1::Point3D,x2::Point3D,ν::Real;Ω_rot= 20e2/Rsun,ℓ_range=nothing)
		
		# We compute δC(x1,x2) = -iΩ ∑ℓ (2ℓ+1)/4π ∂ω (ω^2 P(ω) αℓω*(r₂)αℓω(r₁))∂Δϕ Pl(cos(Δϕ))
		
		Gfn_path = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs dν ν_start_zeros

		if !isnothing(ℓ_range)
			ℓ_range = max(ℓ_range[1],ℓ_arr[1]):min(ℓ_range[end],ℓ_arr[end])
		else
			ℓ_range = ℓ_arr
		end

		Nν = length(ν_arr)

		ν_test_ind = argmin(abs.(ν_arr .- ν))
		ν_on_grid = ν_arr[ν_test_ind]

		# @printf "ν=%.1e ℓmin:%d ℓmax:%d\n" ν_on_grid ℓ_range[1] ℓ_range[end]

		ν_ind_range = max(ν_test_ind-7,1):(ν_test_ind+min(7,ν_test_ind-1))
		ν_match_index = ν_test_ind - ν_ind_range[1] + 1
		dω = 2π*dν

		∂ϕC = ∂ϕ₂Cω(x1,x2,ν_ind_range=ν_ind_range,ℓ_range=ℓ_range)[ν_ind_range .+ ν_start_zeros]

	    ∂ω∂ϕC = D(length(∂ϕC))*∂ϕC ./ dω

		return -im*Ω_rot*∂ω∂ϕC[ν_match_index]
	end

	function δCω_uniform_rotation_rotatedwaves_linearapprox(Δϕ::Real,ν::Real;Ω_rot= 20e2/Rsun,r_obs=Rsun-75e5,ℓ_range=nothing)
		δCω_uniform_rotation_rotatedwaves_linearapprox(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,π/3),ν,Ω_rot=Ω_rot,ℓ_range=ℓ_range)
	end

	function δCω_uniform_rotation_rotatedwaves_linearapprox(n1::Point2D,n2::Point2D,ν::Real;Ω_rot= 20e2/Rsun,r_obs=Rsun-75e5,ℓ_range=nothing)
		δCω_uniform_rotation_rotatedwaves_linearapprox(Point3D(r_obs,n1),Point3D(r_obs,n2),ν,Ω_rot=Ω_rot,ℓ_range=ℓ_range)
	end

	function δCω_uniform_rotation_rotatedwaves_linearapprox(x1::Point3D,x2::Point3D;Ω_rot= 20e2/Rsun,ℓ_range=nothing)
		
		# We compute δC(x1,x2) = -iΩ ∑ℓ (2ℓ+1)/4π ∂ω (ω^2 P(ω) αℓω*(r₂)αℓω(r₁))∂Δϕ Pl(cos(Δϕ))
		
		Gfn_path = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs dν

		if !isnothing(ℓ_range)
			ℓ_range = max(ℓ_range[1],ℓ_arr[1]):min(ℓ_range[end],ℓ_arr[end])
		else
			ℓ_range = ℓ_arr
		end
	
		dω = 2π*dν

		∂ϕC = ∂ϕ₂Cω(x1,x2,ℓ_range=ℓ_range)

	    ∂ω∂ϕC = D(length(∂ϕC))*∂ϕC ./ dω

		return @. -im*Ω_rot*∂ω∂ϕC
	end

	function δCω_uniform_rotation_rotatedwaves_linearapprox(n1::Point2D,n2::Point2D;Ω_rot= 20e2/Rsun,r_obs=Rsun-75e5,ℓ_range=nothing)
		δCω_uniform_rotation_rotatedwaves_linearapprox(Point3D(r_obs,n1),Point3D(r_obs,n2),Ω_rot=Ω_rot,ℓ_range=ℓ_range)
	end

	function δCω_uniform_rotation_rotatedwaves_linearapprox(Δϕ::Real;Ω_rot= 20e2/Rsun,r_obs=Rsun-75e5,ℓ_range=nothing)
		δCω_uniform_rotation_rotatedwaves_linearapprox(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,π/3),Ω_rot=Ω_rot,ℓ_range=ℓ_range)
	end

	function δCω_uniform_rotation_rotatedwaves(Δϕ::Real,ν=3e-3;Ω_rot= 20e2/Rsun,ℓ_range=nothing)
		# We compute δC(Δϕ,ω) = C′(Δϕ,ω) -  C(Δϕ,ω)
		C′(Δϕ,ω) = IFFT(C(m,ω+mΩ),1)
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
			
			tasks_on_proc = Main.parallel_utilities.split_product_across_processors(1:Nν,1:nm,nworkers(),rank)

			C′ωm_arr = zeros(ComplexF64,length(tasks_on_proc))

			m_ind_prev = first(tasks_on_proc)[2]
			Cω_interp_fn = interpolate((ω_arr_full,), C0ωm_arr[:,m_ind_prev], Gridded(Linear()))	

			for (mode_ind,(ω_ind,m_ind)) in enumerate(tasks_on_proc)
				m = m_arr[m_ind]; ω = ω_arr[ω_ind]

				if m_ind != m_ind_prev
					# update the interpolation function
					m_ind_prev = m_ind
					Cω_interp_fn = interpolate((ω_arr_full,), C0ωm_arr[:,m_ind_prev], Gridded(Linear()))
				end
				
				
				C′ωm_arr[mode_ind] =  Cω_interp_fn(ω + m*Ω_rot)
				
			end
			return tasks_on_proc,C′ωm_arr
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

	function δCt_uniform_rotation_rotatedwaves(x1::Point3D,x2::Point3D;Ω_rot = 20e2/Rsun,τ_ind_arr = nothing)
		Gfn_path = Gfn_path_from_source_radius(x1)
		@load "$Gfn_path/parameters.jld2" dν ν_full

		Nν = length(ν_full)
		Nt = 2*(Nν - 1)
		
		C′_t = Cτ_rotating(x1,x2,Ω_rot=Ω_rot,τ_ind_arr=τ_ind_arr)
		τ_ind_arr = axes(C′_t,1)
		C0_t = Ct(x1,x2,dν=dν)[τ_ind_arr]
		return C′_t .- C0_t
	end

	δCt_uniform_rotation_rotatedwaves(n1::Point2D,n2::Point2D;r_obs=Rsun-75e5,Ω_rot= 20e2/Rsun,τ_ind_arr=nothing) = δCt_uniform_rotation_rotatedwaves(Point3D(r_obs,n1),Point3D(r_obs,n2),τ_ind_arr=τ_ind_arr,Ω_rot=Ω_rot)
	δCt_uniform_rotation_rotatedwaves(Δϕ::Real;r_obs=Rsun-75e5,Ω_rot= 20e2/Rsun,τ_ind_arr=nothing) = δCt_uniform_rotation_rotatedwaves(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ),τ_ind_arr=τ_ind_arr,Ω_rot=Ω_rot)

	function δCt_uniform_rotation_rotatedwaves_linearapprox(x1::Point3D,x2::Point3D;Ω_rot = 20e2/Rsun,τ_ind_arr = nothing)
		Gfn_path = Gfn_path_from_source_radius(x1)
		@load "$Gfn_path/parameters.jld2" dν ν_full

		Nν = length(ν_full)
		Nt = 2*(Nν - 1)
		T = 1/dν
		dt = T/Nt
		t = (0:Nt-1).*dt
		
		δCt = -Ω_rot .* t .* ∂ϕ₂Ct(x1,x2,dν=dν)
		if !isnothing(τ_ind_arr)
			return δCt[τ_ind_arr]
		else
			return δCt
		end
	end	

	δCt_uniform_rotation_rotatedwaves_linearapprox(n1::Point2D,n2::Point2D;r_obs=Rsun-75e5,Ω_rot = 20e2/Rsun,τ_ind_arr = nothing) = δCt_uniform_rotation_rotatedwaves_linearapprox(Point3D(r_obs,n1),Point3D(r_obs,n2);Ω_rot = Ω_rot,τ_ind_arr = τ_ind_arr)
	δCt_uniform_rotation_rotatedwaves_linearapprox(Δϕ::Real;r_obs=Rsun-75e5,Ω_rot = 20e2/Rsun,τ_ind_arr = nothing) = δCt_uniform_rotation_rotatedwaves_linearapprox(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ);Ω_rot = Ω_rot,τ_ind_arr = τ_ind_arr)

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


	function h(x1::Point3D,x2::Point3D;plots=false,bounce_no=1)

		Gfn_path = Gfn_path_from_source_radius(x1)
		@load joinpath(Gfn_path,"parameters.jld2") dν ν_full ν_start_zeros ν_arr Nt dt T

		ω_full = 2π.*ν_full

		Cω_x1x2 = Cω(x1,x2)

		C_t = Ct(x1,x2,dν=dν)
		env = abs.(hilbert(C_t))[1:div(Nt,2)] # first half for positive shifts

		# get the time of first bounce arrival
		τ_low,τ_high = bounce_filter(acos(cosχ(x1,x2)),bounce_no)
		τ_low_ind = Int(floor(τ_low/dt)); τ_high_ind = Int(ceil(τ_high/dt))


		# fit a gaussian to obtain the correct bounce
		peak_center = argmax(env[τ_low_ind:τ_high_ind]) + τ_low_ind - 1
		points_around_max = env[peak_center-2:peak_center+2]
		amp = env[peak_center]
		# println("$amp $peak_center")
		# println(points_around_max)
		# Assuming a roughly gaussian peak, fit a quadratic to log(Cω) to obtain σ

		@. model(x,p) = p[1] - (x-p[2])^2/(2p[3]^2)
		p0 = [log(amp),float(peak_center),2.]

		fit = curve_fit(model, peak_center-2:peak_center+2, log.(points_around_max), p0)

		fit_fn = exp.(model(1:div(Nt,2),fit.param))

		σt = fit.param[3]
		t_inds_range = Int(floor(fit.param[2] - 2σt)):Int(ceil(fit.param[2] + 2σt))
		f_t = zeros(Nt)
		f_t[t_inds_range] .= 1

		∂tCt = irfft(rfft(C_t).*im.*ω_full,Nt)
		h_t = - f_t .* ∂tCt ./ sum(f_t.*∂tCt.^2 .* dt)
		h_ω = rfft(h_t).*dt

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

	h(n1::Point2D,n2::Point2D,r_obs::Real=Rsun-75e5;plots=false,bounce_no=1) = h(Point3D(r_obs,n1),Point3D(r_obs,n2),plots=plots,bounce_no=bounce_no)
	h(Δϕ::Real,r_obs::Real=Rsun-75e5;plots=false,bounce_no=1) = h(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ),plots=plots,bounce_no=bounce_no)

	
end

module kernel

	using Reexport
	using Main.crosscov
	using PyCall
	@pyimport scipy.integrate as integrate

	function kernel_uniform_rotation_uplus(n1::Point2D,n2::Point2D,r_obs::Real=Rsun-75e5;ℓ_range=nothing)
		Gfn_path = Gfn_path_from_source_radius(r_obs)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr ℓ_arr num_procs dν ν_start_zeros

		if isnothing(ℓ_range)
			ℓ_range = ℓ_arr
		else
			ℓ_range = max(ℓ_range[1],ℓ_arr[1]):min(ℓ_range[end],ℓ_arr[end])
		end

		Base.GC.gc()

		h_ω_arr = h(n1,n2,r_obs).h_ω[ν_start_zeros .+ (1:length(ν_arr))] # only in range

		Base.GC.gc()

		r_obs_ind = argmin(abs.(r .- r_obs))

		dω = dν*2π

		Nν = length(ν_arr)

		function kernel_uniform_rotation_uplus_somemodes(rank)
		
			ℓ_ωind_iter_on_proc = collect(split_product_across_processors(ℓ_arr,1:Nν,nworkers(),rank))
			proc_id_range = get_processor_range_from_split_array(ℓ_arr,1:Nν,ℓ_ωind_iter_on_proc,num_procs)

			∂ϕ₂Pl_cosχ = dPl(cosχ(n1,n2),ℓmax=ℓ_arr[end]).*∂ϕ₂cosχ(n1,n2)

			K = zeros(ComplexF64,nr)

			for proc_id in proc_id_range
				G_proc_file = joinpath(Gfn_path,@sprintf "Gfn_proc_%03d.jld2" proc_id)
				@load G_proc_file Gfn_arr
		    	for Gfn in Gfn_arr
		    		ℓ = Gfn.mode.ℓ
		    		(ℓ==0) && continue
		    		ω = Gfn.mode.ω
		    		ω_ind = Gfn.mode.ω_ind
		    		
		    		((ℓ,ω_ind) ∉ ℓ_ωind_iter_on_proc) && continue

				    G0 = Gfn.G[:,0,0]
				    G1 = Gfn.G[:,1,0]
				    
				    K .+= @. dω/2π * ω^3 * Powspec(ω) * (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] * imag(h_ω_arr[ω_ind]) * real(
				    				conj(G0[r_obs_ind])*(G0^2 -  2G0*G1/Ω(ℓ,0) + G1^2*(ℓ*(ℓ+1)-1)/Ω(ℓ,0)^2))
				end
			end

			return 8*√(3/4π)*im .* K .* r .* ρ
		end

		kernel_futures = [@spawnat p kernel_uniform_rotation_uplus_somemodes(rank) for (rank,p) in enumerate(workers_active(ℓ_arr,1:Nν))]

		K = sum(fetch.(kernel_futures))
	end

	function δτ_uniform_rotation_firstborn_int_K_u(n1::Point2D,n2::Point2D,r_obs::Real=Rsun-75e5;Ω_rot=20e2/Rsun)
		K₊ = kernel_uniform_rotation_uplus(n1,n2,r_obs)
		u⁺ = @. √(4π/3)*im*Ω_rot*r

		δτ = real(integrate.simps(K₊.*u⁺,x=r))
	end

	function δτ_uniform_rotation_firstborn_int_hω_δCω(n1::Point2D,n2::Point2D,r_obs::Real=Rsun-75e5;Ω_rot=20e2/Rsun)
		Gfn_path = Gfn_path_from_source_radius(r_obs)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr dν ν_start_zeros

		Base.GC.gc()

		h_ω_arr = h(n1,n2,r_obs).h_ω[ν_start_zeros .+ (1:length(ν_arr))] # only in range

		r_obs_ind = argmin(abs.(r .- r_obs))

		dω = dν*2π

		Nν = length(ν_arr)

		δτ = zero(Float64)

		δC = δCω_uniform_rotation_firstborn_integrated_over_angle(n1,n2,r_obs=r_obs,Ω_rot=Ω_rot)

		for (hω,δCω) in zip(h_ω_arr,δC)
			δτ += dω/2π * 2real(conj(hω)*δCω)
		end

		return δτ
	end	

	function δτ_uniform_rotation_rotatedframe_int_hω_δCω_linearapprox(n1::Point2D,n2::Point2D,r_obs::Real=Rsun-75e5;Ω_rot=20e2/Rsun)
		Gfn_path = Gfn_path_from_source_radius(r_obs)
		@load joinpath(Gfn_path,"parameters.jld2") ν_arr dν ν_start_zeros

		Nν = length(ν_arr)

		Base.GC.gc()

		h_ω_arr = h(n1,n2,r_obs).h_ω[ν_start_zeros .+ (1:Nν)] # only in range

		r_obs_ind = argmin(abs.(r .- r_obs))

		dω = dν*2π

		δτ = zero(Float64)

		δC = Main.crosscov.δCω_uniform_rotation_rotatedwaves_linearapprox(n1,n2,r_obs=r_obs,Ω_rot=Ω_rot)[ν_start_zeros .+ (1:Nν)]

		for (hω,δCω) in zip(h_ω_arr,δC)
			δτ += dω/2π * 2real(conj(hω)*δCω)
		end

		return δτ
	end

	function δτ_uniform_rotation_rotatedframe_int_ht_δCt(n1::Point2D,n2::Point2D;Ω_rot=20e2/Rsun,r_obs=Rsun-75e5)
		Gfn_path = Gfn_path_from_source_radius(r_obs)
		@load joinpath(Gfn_path,"parameters.jld2") dν

		T = 1/dν

		Base.GC.gc()

		h_arr = h(n1,n2,r_obs)
		h_t = h_arr.h_t
		τ_ind_arr = h_arr.t_inds_range

		Nt = size(h_t,1)
		dt = T/Nt

		δC_t = δCt_uniform_rotation_rotatedwaves(n1,n2;Ω_rot=Ω_rot,τ_ind_arr=τ_ind_arr,r_obs=r_obs)

		δτ = integrate.simps(h_t[τ_ind_arr].*δC_t.parent,dx=dt)
	end

	function δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1::Point2D,n2::Point2D;Ω_rot=20e2/Rsun,r_obs=Rsun-75e5)
		Gfn_path = Gfn_path_from_source_radius(r_obs)
		@load joinpath(Gfn_path,"parameters.jld2") dν

		T = 1/dν

		Base.GC.gc()

		h_arr = h(n1,n2,r_obs)
		h_t = h_arr.h_t
		τ_ind_arr = h_arr.t_inds_range

		Nt = size(h_t,1)
		dt = T/Nt

		δC_t = Main.crosscov.δCt_uniform_rotation_rotatedwaves_linearapprox(n1,n2;Ω_rot=Ω_rot,τ_ind_arr=τ_ind_arr,r_obs=r_obs)

		δτ = integrate.simps(h_t[τ_ind_arr].*δC_t,dx=dt)
	end
end