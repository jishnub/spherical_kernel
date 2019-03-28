include("./crosscov.jl")

#################################################################
# Kernels and travel times
#################################################################

module kernel

	using Reexport
	@reexport using Main.crosscov
	@pyimport scipy.integrate as integrate
	import WignerSymbols: clebschgordan
	using Libdl, WignerD, FileIO

	################################################################################################################
	# Validation for uniform rotation
	################################################################################################################

	uniform_rotation_uplus(Ω_rot=20e2/Rsun) = @. √(4π/3)*im*Ω_rot*r

	function kernel_uniform_rotation_uplus(x1::Point3D,x2::Point3D;
		ℓ_range=nothing,ν_ind_range=nothing,bounce_no=1,kwargs...)
		
		r_src = get(kwargs,:r_src,Rsun-75e5) :: Float64

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_x1 = Gfn_path_from_source_radius(x1)
		Gfn_path_x2 = Gfn_path_from_source_radius(x2)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs dν ν_start_zeros dω Nν_Gfn

		num_procs_obs1 = load(joinpath(Gfn_path_x1,"parameters.jld2"),"num_procs")
		num_procs_obs2 = load(joinpath(Gfn_path_x2,"parameters.jld2"),"num_procs")

		ℓ_range = intersect_fallback(ℓ_range,ℓ_arr)
		ν_ind_range = intersect_fallback(ν_ind_range,1:Nν_Gfn)

		h_ω = get(kwargs,:hω,nothing)
		if isnothing(h_ω)
			h_ω = hω(x1,x2;bounce_no=bounce_no,kwargs...)
		end
		h_ω = h_ω[ν_start_zeros .+ ν_ind_range]
		# h_ω_arr = hω(x1,x2,bounce_no=bounce_no,ℓ_range=ℓ_range,r_src=r_src)[ν_start_zeros .+ ν_ind_range] # only in range

		r₁_ind = argmin(abs.(r .- x1.r))
		r₂_ind = argmin(abs.(r .- x2.r))

		function summodes(rank,rank_node,np_node,channel_on_node)
		
			ℓ_ωind_iter_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank)
			proc_id_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)

			∂ϕ₂Pl_cosχ = dPl(cosχ(x1,x2),ℓmax=ℓ_arr[end]).*∂ϕ₂cosχ(x1,x2)

			K = zeros(ComplexF64,nr)

			δG_r₁_rsrc = zeros(ComplexF64,nr) # This is not actually δG, it's just a radial function dependent on f

			G = zeros(nr,2,2)
			Gsrc = zeros(ComplexF64,1:nr,0:1)
			Gobs1 = zeros(ComplexF64,1:nr,0:1)

			δG_r₂_rsrc = zeros(ComplexF64,nr)
			Gobs2 = zeros(ComplexF64,1:nr,0:1)			

			for proc_id in proc_id_range

				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")				
				# Get a list of (ℓ,ω) in this file
				modes_in_Gsrc_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_Gsrc_file)

					# Get index of this mode in the file
					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_Gsrc_file,(ℓ,ω_ind))
					ω = dω*(ν_start_zeros + ω_ind)

					# Green function about source location
		    		G .= read(Gsrc_file[1],:,:,1:2,1,1,ℓω_index_Gsrc_file)
		    		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		G_r₁_rsrc = Gsrc[r₁_ind,0]
		    		G_r₂_rsrc = Gsrc[r₂_ind,0]

		    		# Green function about receiver location
		    		proc_id_mode_Gobs1,ℓω_index_Gobs1_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs1)
		    		Gobs1_file = FITS(joinpath(Gfn_path_x1,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs1),"r")

		    		G .= read(Gobs1_file[1],:,:,1:2,1,1,ℓω_index_Gobs1_file)
		    		@. Gobs1[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gobs1[:,1] = G[:,1,2] + im*G[:,2,2]

		    		@. δG_r₁_rsrc = Gsrc[:,0] * Gobs1[:,0] - Gsrc[:,0] * Gobs1[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs1[:,0] + (ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs1[:,1]/Ω(ℓ,0)

					close(Gobs1_file)

				    if r₁_ind != r₂_ind

				    	proc_id_mode_Gobs2,ℓω_index_Gobs2_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs2)

				    	Gobs2_file = FITS(joinpath(Gfn_path_x2,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs2),"r")
			    		G .= read(Gobs2_file[1],:,:,1:2,1,1,ℓω_index_Gobs2_file)
			    		@. Gobs2[:,0] = G[:,1,1] + im*G[:,2,1]
			    		@. Gobs2[:,1] = G[:,1,2] + im*G[:,2,2]

			    		@. δG_r₂_rsrc = Gsrc[:,0] * Gobs2[:,0] - Gsrc[:,0] * Gobs2[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs2[:,0] + 
		    						(ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs2[:,1]/Ω(ℓ,0)

					    close(Gobs2_file)
					else
					    @. δG_r₂_rsrc = δG_r₁_rsrc
					end

					@. K +=  dω/2π * ω^3 * Powspec(ω) * (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] * imag(h_ω[ω_ind]) * 
								(conj(δG_r₁_rsrc)*G_r₂_rsrc + conj(G_r₁_rsrc)*δG_r₂_rsrc) 

				end

				close(Gsrc_file)
			end

			@. K *= 4*√(3/4π)*im * r * ρ

			if rank_node == 0
				for n in 1:np_node-1
					K .+= take!(channel_on_node)
				end
				put!(channel_on_node,K)
			else
				put!(channel_on_node,K)
			end
			return nothing
		end

		procs_used = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs_used)

		pmapsum(Vector{ComplexF64},summodes,procs_used)
	end

	function kernel_uniform_rotation_uplus(n1::Point2D,n2::Point2D;
		ℓ_range=nothing,ν_ind_range=nothing,bounce_no=1,kwargs...)

		r_src = get(kwargs,:r_src,Rsun-75e5) :: Float64
		r_obs = get(kwargs,:r_src,Rsun-75e5) :: Float64

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_obs = Gfn_path_from_source_radius(r_obs)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs dν dω ν_start_zeros Nν_Gfn

		num_procs_obs = load(joinpath(Gfn_path_obs,"parameters.jld2"),"num_procs")

		ℓ_range = intersect_fallback(ℓ_range,ℓ_arr)
		ν_ind_range = intersect_fallback(ν_ind_range,1:Nν_Gfn)

		r_obs_ind = argmin(abs.(r .- r_obs))

		∂ϕ₂Pl_cosχ = dPl(cosχ(n1,n2),ℓmax=ℓ_arr[end]).*∂ϕ₂cosχ(n1,n2)
		
		h_ω = get(kwargs,:hω,nothing)
		if isnothing(h_ω)
			h_ω = hω(n1,n2;bounce_no=bounce_no,kwargs...)
		end
		h_ω = h_ω[ν_start_zeros .+ ν_ind_range]

		# h_ω = hω(n1,n2,bounce_no=bounce_no,ℓ_range=ℓ_range,r_src=r_src,r_obs=r_obs) # only in range

		function summodes(rank,rank_node,np_node,channel_on_node)
		
			ℓ_ωind_iter_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank)
			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			proc_id_range_Gobs = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs_obs)

			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)
			Gfn_fits_files_obs = Gfn_fits_files(Gfn_path_obs,proc_id_range_Gobs)

			K = zeros(ComplexF64,nr)

			δG_robs_rsrc = zeros(ComplexF64,nr) # This is not actually δG, it's just a radial function dependent on f

			G = zeros(nr,2,2)
			Gsrc = zeros(ComplexF64,nr,0:1)
			Gobs = zeros(ComplexF64,nr,0:1)

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_Gsrc_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_Gsrc_file)

					# Get index of this mode in the file
					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_Gsrc_file,(ℓ,ω_ind))
					ω = dω*(ν_start_zeros + ω_ind)

					# Green function about source location
		    		G .= read(Gsrc_file[1],:,:,1:2,1,1,ℓω_index_Gsrc_file)
		    		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		G_robs_rsrc = Gsrc[r_obs_ind,0]

		    		# Green function about receiver location
		    		
		    		proc_id_mode_Gobs,ℓω_index_Gobs_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs)
		    		Gobs_file = Gfn_fits_files_obs[proc_id_mode_Gobs]

		    		G .= read(Gobs_file[1],:,:,1:2,1,1,ℓω_index_Gobs_file)
		    		@. Gobs[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gobs[:,1] = G[:,1,2] + im*G[:,2,2]

		    		@. δG_robs_rsrc = Gsrc[:,0] * Gobs[:,0] - Gsrc[:,0] * Gobs[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs[:,0] + 
		    						(ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs[:,1]/Ω(ℓ,0)


					@. K +=  dω/2π * ω^3 * Powspec(ω) * 
							(2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] * imag(h_ω[ω_ind]) * 
								2real(conj(δG_robs_rsrc)*G_robs_rsrc) 

				end
			end

			close.(values(Gfn_fits_files_src))
			close.(values(Gfn_fits_files_obs))

			@. K *= 4*√(3/4π)*im * r * ρ

			if rank_node == 0
				for n in 1:np_node-1
					K .+= take!(channel_on_node)
				end
				put!(channel_on_node,K)
			else
				put!(channel_on_node,K)
			end
			return nothing
		end

		procs_used = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs_used)

		pmapsum(Vector{ComplexF64},summodes,procs_used)
	end
	
	function kernel_uniform_rotation_uplus(n1::Point2D,n2_arr::Vector{<:Point2D};
		Cω_arr=Array{Nothing,2}(undef,1,length(n2_arr)),
		hω_arr=nothing,
		ℓ_range=nothing,ν_ind_range=nothing,bounce_no=1,r_src=Rsun-75e5,r_obs=Rsun-75e5,kwargs...)

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_obs = Gfn_path_from_source_radius(r_obs)

		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs dω ν_start_zeros

		num_procs_obs = load(joinpath(Gfn_path_obs,"parameters.jld2"),"num_procs")

		ℓ_range = intersect_fallback(ℓ_range,ℓ_arr)
		ν_ind_range = intersect_fallback(ν_ind_range,1:Nν_Gfn)

		r_obs_ind = argmin(abs.(r .- r_obs))

		function summodes(rank,rank_node,np_node,channel_on_node)
		
			ℓ_ωind_iter_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank)
			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			proc_id_range_Gobs = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs_obs)

			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)
			Gfn_fits_files_obs = Gfn_fits_files(Gfn_path_obs,proc_id_range_Gobs)

			K = zeros(ComplexF64,nr,length(n2_arr))

			δG_robs_rsrc = zeros(ComplexF64,nr) # This is not actually δG, it's just a radial function dependent on f

			G = zeros(nr,2,2)
			Gsrc = zeros(ComplexF64,nr,0:1)
			Gobs = zeros(ComplexF64,nr,0:1)

			fr = zeros(ComplexF64,nr)

			∂ϕ₂Pl_cosχ = zeros(0:ℓ_arr[end])

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_Gsrc_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_Gsrc_file)

					# Get index of this mode in the file
					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_Gsrc_file,(ℓ,ω_ind))
					ω = dω*(ν_start_zeros + ω_ind)

					# Green function about source location
		    		G .= read(Gsrc_file[1],:,:,1:2,1,1,ℓω_index_Gsrc_file)
		    		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		G_robs_rsrc = Gsrc[r_obs_ind,0]

		    		# Green function about receiver location
		    		proc_id_mode_Gobs,ℓω_index_Gobs_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,
		    												(ℓ,ω_ind),num_procs_obs)
		    		Gobs_file = Gfn_fits_files_obs[proc_id_mode_Gobs]

		    		G .= read(Gobs_file[1],:,:,1:2,1,1,ℓω_index_Gobs_file)
		    		@. Gobs[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gobs[:,1] = G[:,1,2] + im*G[:,2,2]

		    		@. δG_robs_rsrc = Gsrc[:,0] * Gobs[:,0] - Gsrc[:,0] * Gobs[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs[:,0] + 
		    						(ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs[:,1]/Ω(ℓ,0)

		    		@. fr = dω/2π * ω^3 * Powspec(ω) * 2real(conj(δG_robs_rsrc)*G_robs_rsrc) 

					for n2ind in 1:length(n2_arr)
						@. K[:,n2ind] +=   fr * (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ_arr[n2ind,ℓ] * 
											imag(hω_arr[n2ind,ω_ind])
					end
					

				end
			end

			close.(values(Gfn_fits_files_src))
			close.(values(Gfn_fits_files_obs))

			@. K *= 4*√(3/4π)*im * r * ρ

			if rank_node == 0
				for n in 1:np_node-1
					K .+= take!(channel_on_node)
				end
				put!(channel_on_node,K)
			else
				put!(channel_on_node,K)
			end
			put!(tracker,0)
			return nothing
		end

		∂ϕ₂Pl_cosχ_arr = zeros(0:ℓ_arr[end],length(n2_arr))
		for (n2ind,n2) in enumerate(n2_arr)
			∂ϕ₂Pl_cosχ_arr[:,n2ind] = dPl(cosχ(n1,n2),ℓmax=ℓ_arr[end]).*∂ϕ₂cosχ(n1,n2)
		end
		∂ϕ₂Pl_cosχ_arr = copy(transpose(∂ϕ₂Pl_cosχ_arr))
		
		
		if  isnothing(hω_arr)
			hω_arr = zeros(ν_ind_range,length(n2_arr))
			τ_ind_arr = get(kwargs,:τ_ind_arr,nothing)
			for (n2ind,n2) in enumerate(n2_arr)
				hω_arr[ν_ind_range,n2ind] = hω(Cω_arr[:,n2ind],n1,n2,bounce_no=bounce_no,
				ℓ_range=ℓ_range,τ_ind_arr=τ_ind_arr[n2ind],
				r_src=r_src,r_obs=r_obs)[ν_start_zeros .+ ν_ind_range]
			end
			hω_arr = copy(transpose(hω_arr)) :: Array{ComplexF64,2}
		else
			hω_arr = copy(transpose(hω_arr))[:,ν_start_zeros .+ ν_ind_range] :: Array{ComplexF64,2}
		end

		procs_used = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs_used)

		tracker = RemoteChannel(()->Channel{Int64}(100),1)

		prog_bar = Progress(num_workers,1,"First born travel times : ")

		K₊ = zeros(ComplexF64,nr,length(n2_arr))
		@sync begin
			@async K₊ .= pmapsum(Array{ComplexF64,2},summodes,procs_used)
			@async for n in 1:num_workers
				take!(tracker)
				next!(prog_bar)
			end
		end

		close(tracker)
		return K₊
	end

	function δτ_uniform_rotation_firstborn_int_K_u(x1,x2;Ω_rot=20e2/Rsun,kwargs...)
		K₊ = kernel_uniform_rotation_uplus(x1,x2;kwargs...)
		u⁺ = uniform_rotation_uplus(Ω_rot)

		δτ = real.(integrate.simps(K₊.*u⁺,x=r,axis=0))
	end

	function δτ_uniform_rotation_firstborn_int_hω_δCω(n1,n2;Ω_rot=20e2/Rsun,bounce_no=1,kwargs...)
		r_src = get(kwargs,:r_src,Rsun-75e5) :: Float64
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dω ν_start_zeros Nν_Gfn

		h_ω = get(kwargs,:hω,nothing)
		if isnothing(h_ω)
			h_ω = hω(n1,n2;bounce_no=bounce_no,kwargs...)
		end
		h_ω = h_ω[ν_start_zeros .+ (1:Nν_Gfn)] # only in range

		δCω = δCω_uniform_rotation_firstborn_integrated_over_angle(n1,n2;
			Ω_rot=Ω_rot,kwargs...)[ν_start_zeros .+ (1:Nν_Gfn)]

		δτ = sum(@. dω/2π * 2real(conj(h_ω)*δCω))
	end

	function δτ_uniform_rotation_rotatedframe_int_hω_δCω_linearapprox(n1::Point2D,n2::Point2D;
		Ω_rot=20e2/Rsun,bounce_no=1,kwargs...)

		r_src = get(kwargs,:r_src,Rsun-75e5) :: Float64
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dω ν_start_zeros Nν_Gfn

		h_ω = get(kwargs,:hω,nothing)
		if isnothing(h_ω)
			h_ω = hω(n1,n2;bounce_no=bounce_no,kwargs...)
		end
		h_ω = h_ω[ν_start_zeros .+ (1:Nν_Gfn)] # only in range

		δCω = δCω_uniform_rotation_rotatedwaves_linearapprox(n1,n2;
			Ω_rot=Ω_rot,kwargs...)[ν_start_zeros .+ (1:Nν_Gfn)]

		δτ = sum(@. dω/2π * 2real(conj(h_ω)*δCω))
	end

	function δτ_uniform_rotation_rotatedframe_int_ht_δCt(n1::Point2D,n2::Point2D;
		Ω_rot=20e2/Rsun,bounce_no=1,kwargs...)

		r_src = get(kwargs,:r_src,Rsun-75e5) :: Float64
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt Nt dν

		Cω_n1n2 = Cω(n1,n2;kwargs...)
		Ct_n1n2 = Ct(Cω_n1n2,dν)
		τ_ind_arr = time_window_indices_by_fitting_bounce_peak(Ct_n1n2,n1,n2,dt=dt,Nt=Nt,bounce_no=bounce_no)

		h_t = get(kwargs,:ht,nothing)
		if isnothing(h_t)
			h_ω = get(kwargs,:hω,nothing)
			if isnothing(h_ω)
				h_t = ht(Cω_n1n2,n1,n2;bounce_no=bounce_no,kwargs...)
			else
				h_t = @fft_ω_to_t(h_ω)
			end
		end
		h_t = h_t[τ_ind_arr]

		δC_t = δCt_uniform_rotation_rotatedwaves(n1,n2;Ω_rot=Ω_rot,τ_ind_arr=τ_ind_arr,kwargs...)

		δτ = sum(h_t.*δC_t.parent)*dt
	end

	function δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1::Point2D,n2::Point2D;
		Ω_rot=20e2/Rsun,bounce_no=1,τ_ind_arr=nothing,kwargs...)

		r_src = get(kwargs,:r_src,Rsun-75e5) :: Float64
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt dν Nt

		if isnothing(get(kwargs,:Cω,nothing)) && isnothing(get(kwargs,:∂ϕ₂Cω,nothing))
			Cω_n1n2,∂ϕ₂Cω_n1n2 = Cω_∂ϕ₂Cω(n1,n2;kwargs...)
		elseif isnothing(get(kwargs,:Cω,nothing))
			Cω_n1n2 = Cω(n1,n2;kwargs...)
		elseif isnothing(get(kwargs,:∂ϕ₂Cω,nothing))
			∂ϕ₂Cω_n1n2 = ∂ϕ₂Cω(n1,n2;kwargs...)
		end
		
		if isnothing(τ_ind_arr)
			Ct_n1n2 = Ct(Cω_n1n2,dν)
			τ_ind_arr = time_window_indices_by_fitting_bounce_peak(Ct_n1n2,
								n1,n2,dt=dt,Nt=Nt,bounce_no=bounce_no)
		end

		if isnothing(get(kwargs,:ht,nothing))
			if isnothing(get(kwargs,:hω,nothing))
				h_t = ht(Cω_n1n2,n1,n2;τ_ind_arr=τ_ind_arr,kwargs...)
			else
				h_t = @fft_ω_to_t(h_ω)	
			end
		end

		∂ϕ₂Ct_n1n2 = ∂ϕ₂Ct(∂ϕ₂Cω_n1n2,dν)
		δC_t = δCt_uniform_rotation_rotatedwaves_linearapprox(∂ϕ₂Ct_n1n2;
											Ω_rot=Ω_rot,kwargs...)
		
		δτ = sum(h_t[τ_ind_arr].*δC_t[τ_ind_arr])*dt
	end

	function δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1::Point2D,n2_arr::Vector{<:Point2D};
		Ω_rot=20e2/Rsun,bounce_no=1,τ_ind_arr=nothing,kwargs...)

		r_src = get(kwargs,:r_src,Rsun-75e5) :: Float64
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt dν Nt

		Cω_n1n2 = get(kwargs,:Cω,nothing)
		∂ϕ₂Cω_n1n2 = get(kwargs,:∂ϕ₂Cω,nothing)

		if isnothing(Cω_n1n2) && isnothing(∂ϕ₂Cω_n1n2)
			Cω_n1n2,∂ϕ₂Cω_n1n2 = Cω_∂ϕ₂Cω(n1,n2_arr;kwargs...)
		elseif isnothing(Cω_n1n2)
			Cω_n1n2 = Cω(n1,n2_arr;kwargs...)
		elseif isnothing(∂ϕ₂Cω_n1n2)
			∂ϕ₂Cω_n1n2 = ∂ϕ₂Cω(n1,n2_arr;kwargs...)
		end
		
		if isnothing(τ_ind_arr)
			Ct_n1n2 = Ct(Cω_n1n2,dν)
			τ_ind_arr = time_window_indices_by_fitting_bounce_peak(Ct_n1n2,
								n1,n2_arr,dt=dt,Nt=Nt,bounce_no=bounce_no)
		end

		h_t = get(kwargs,:ht,nothing)
		if isnothing(h_t)
			h_ω = get(kwargs,:hω,nothing)
			if isnothing(h_ω)
				h_t = ht(Cω_n1n2,n1,n2_arr;τ_ind_arr=τ_ind_arr,kwargs...) 
			else
				h_t = @fft_ω_to_t(h_ω)	
			end
		end

		∂ϕ₂Ct_n1n2 = ∂ϕ₂Ct(∂ϕ₂Cω_n1n2,dν)
		δC_t = δCt_uniform_rotation_rotatedwaves_linearapprox(∂ϕ₂Ct_n1n2;
											Ω_rot=Ω_rot,kwargs...)
		
		δτ = [sum(h_t[τ_ind_arr[n2ind],n2ind].*δC_t[τ_ind_arr[n2ind],n2ind])*dt 
				for n2ind in 1:length(n2_arr)]
	end

	function traveltimes_validate(n1,n2;kwargs...)

		r_src = get(kwargs,:r_src,Rsun-75e5) :: Float64
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt dν

		h_t = ht(n1,n2;kwargs...)
		h_ω = @fft_t_to_ω(h_t)

		δτ1 = δτ_uniform_rotation_firstborn_int_K_u(n1,n2;hω=h_ω,kwargs...)
		@printf "%-50s %g\n" "First Born, ∫dr u(r) K(r)" round(δτ1,sigdigits=3)

		δτ2 = δτ_uniform_rotation_firstborn_int_hω_δCω(n1,n2;hω=h_ω,kwargs...)
		@printf "%-50s %g\n" "First Born, ∫dω/2π h(ω) δC_FB(ω)" round(δτ2,sigdigits=3)

		δτ3 = δτ_uniform_rotation_rotatedframe_int_hω_δCω_linearapprox(n1,n2;hω=h_ω,kwargs...)
		@printf "%-50s %g\n" "Rotated frame, ∫dω/2π h(ω) δC_R_lin(ω)" round(δτ3,sigdigits=3)

		δτ4 = δτ_uniform_rotation_rotatedframe_int_ht_δCt(n1,n2;ht=h_t,kwargs...)
		@printf "%-50s %g\n" "Rotated frame, ∫dt h(t) δC_R(t)" round(δτ4,sigdigits=3)

		δτ5 = δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1,n2;ht=h_t,kwargs...)
		@printf "%-50s %g\n" "Rotated frame, ∫dt h(t) δC_R_lin(t)" round(δτ5,sigdigits=3)
	end

	################################################################################################################
	# All kernels
	################################################################################################################
	
	Nℓ′ℓs(ℓ′,ℓ,s) = √((2ℓ+1)*(2ℓ′+1)/(4π*(2s+1)))

	function flow_axisymmetric_srange(x1::Point3D,x2::Point3D,s_max;
		ℓ_range=nothing,ν_ind_range=nothing,bounce_no=1,r_src=Rsun-75e5,K_components=-1:1)

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_x1 = Gfn_path_from_source_radius(x1)
		Gfn_path_x2 = Gfn_path_from_source_radius(x2)

		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs dν ν_start_zeros

		num_procs_x1 = load(joinpath(Gfn_path_x1,"parameters.jld2"),"num_procs")
		num_procs_x2 = load(joinpath(Gfn_path_x2,"parameters.jld2"),"num_procs")

		ℓ_range = intersect_fallback(ℓ_range,ℓ_arr)
		ν_ind_range = intersect_fallback(ν_ind_range,1:Nν_Gfn)
		
		h_ω_arr = h(x1,x2,bounce_no=bounce_no,ℓ_range=ℓ_range,r_src=r_src).h_ω[ν_start_zeros .+ ν_ind_range] # only in range

		r₁_ind = argmin(abs.(r .- x1.r))
		r₂_ind = argmin(abs.(r .- x2.r))

		obs_at_same_height = r₁_ind == r₂_ind

		dω = dν*2π

		function summodes(rank,rank_node,np_node,channel_on_node)
		
			ℓ_ωind_iter_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank)
			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			# Get a list of all modes that will be accessed.
			# This can be used to open the fits files before the loops begin.
			# This will cut down on FITS IO costs

			# Gℓ′ω(r,robs) files
			first_mode = first(ℓ_ωind_iter_on_proc)
			last_mode = last(collect(ℓ_ωind_iter_on_proc))
			ℓ′_min_first_mode = max(minimum(ℓ_arr),abs(first_mode[1]-s_max))
			ℓ′_max_last_mode = min(maximum(ℓ_arr),last_mode[1]+s_max)
			modes_minmax = minmax_from_split_array(ℓ_ωind_iter_on_proc)
			ℓ_max_proc = modes_minmax.ℓ_max
			ℓ′_max_proc =  ℓ_max_proc + s_max
			
			proc_id_min_G_x1 = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,
									(ℓ′_min_first_mode,first_mode[2]),num_procs_x1)
			proc_id_max_G_x1 = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,
									(ℓ′_max_last_mode,last_mode[2]),num_procs_x1)
			if num_procs_x1 != num_procs_x2
				proc_id_min_G_x2 = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,
										(ℓ′_min_first_mode,first_mode[2]),num_procs_x2)
				proc_id_max_G_x2 = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,
										(ℓ′_max_last_mode,last_mode[2]),num_procs_x2)
			else
				proc_id_min_G_x2 = proc_id_min_G_x1
				proc_id_max_G_x2 = proc_id_max_G_x1
			end

			Gfn_fits_files_obs1 = Gfn_fits_files(Gfn_path_x1,proc_id_min_G_x1:proc_id_max_G_x1)
			Gfn_fits_files_obs2 = Gfn_fits_files(Gfn_path_x2,proc_id_min_G_x2:proc_id_max_G_x2)

			K = zeros(nr,minimum(K_components):maximum(K_components),1:s_max)

			Gsrc = zeros(ComplexF64,nr,0:1)
			drGsrc = zeros(ComplexF64,nr,0:1)
			G_x1 = zeros(ComplexF64,nr,0:1)
			G_x2 = zeros(ComplexF64,nr,0:1)

			# temporary array to precompute the radial part, indices are (r,η)
			f_radial_0_r₁ = zeros(ComplexF64,nr,0:1) # the η=-1 and η=1 terms are identical
			f_radial_1_r₁ = zeros(ComplexF64,nr,-1:1)
			fℓ′ℓsω_r₁ = zeros(ComplexF64,nr,0:1)

			f_radial_0_r₂ = zeros(ComplexF64,nr,0:1) # the η=-1 and η=1 terms are identical
			f_radial_1_r₂ = zeros(ComplexF64,nr,-1:1)
			fℓ′ℓsω_r₂ = zeros(ComplexF64,nr,0:1)

			# Clebsch Gordan coefficients, indices are (s,η,t)
			Cℓ′ℓ = zeros(-1:1,1:s_max,-1:0)

			# cache Ylmn arrays to speed up computation of BiPoSH_s0
			# Create an OffsetArray of the Ylmn OffsetArrays
			T = OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
			Yℓ′_n1_arr = OffsetArray{T}(undef,-s_max:s_max)
    		Yℓ′_n2_arr = OffsetArray{T}(undef,-s_max:s_max)
    		# The array indices will be ℓ′-ℓ values to ensure that we can 
    		# overwrite the preallocated arrays
    		
    		for ind in eachindex(Yℓ′_n1_arr)
    			Yℓ′_n1_arr[ind] = zeros(ComplexF64,-ℓ′_max_proc:ℓ′_max_proc,0:0)
    		end

    		for ind in eachindex(Yℓ′_n2_arr)
    			Yℓ′_n2_arr[ind] = zeros(ComplexF64,-ℓ′_max_proc:ℓ′_max_proc,0:0)
    		end

			# Cache bipolar spherical harmonics in a dict on each worker
			Y12 = Dict{NTuple{2,Int64},OffsetArray{ComplexF64,3,Array{ComplexF64,3}}}()
			Y21 = Dict{NTuple{2,Int64},OffsetArray{ComplexF64,3,Array{ComplexF64,3}}}()

			# keep track of ℓ to cache Yℓ′ by rolling arrays
			# if ℓ changes by 1 arrays can be rolled
			# if the ℓ wraps back then δℓ will be negative. 
			# In this case we need to recompute the Yℓ′ arrays
			ℓ_prev = first_mode[1]

			# Loop over the Greenfn files
			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_src_Gfn_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_src_Gfn_file)

					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_src_Gfn_file,(ℓ,ω_ind))

					ω = dω*(ν_start_zeros + ω_ind)
		    		
		    		G = read(Gsrc_file[1],:,:,1:2,1,1,ℓω_index_Gsrc_file)

		    		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		G_r₁_rsrc = Gsrc[r₁_ind,0]
		    		G_r₂_rsrc = Gsrc[r₂_ind,0]

		    		if 0 in K_components
		    			G = read(Gsrc_file[1],:,:,1:2,1,2,ℓω_index_Gsrc_file)
			    		@. drGsrc[:,0] = G[:,1,1] + im*G[:,2,1]
			    		@. drGsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		end

		    		G = nothing

		    		ℓ′_range = intersect(ℓ_arr,abs(ℓ-s_max):ℓ+s_max)

		    		# Precompute Ylmatrix to speed up evaluation of BiPoSH_s0
		    		Yℓ_n2 = Ylmatrix(ℓ,x2,n_range=0:0)
		    		Yℓ_n1 = Ylmatrix(ℓ,x1,n_range=0:0)

		    		if (ℓ - ℓ_prev) == 1
			    		# roll the Yℓ′ arrays
			    		for ind in first(ℓ′_range)-ℓ:last(ℓ′_range)-ℓ-1
			    			@. Yℓ′_n1_arr[ind] = Yℓ′_n1_arr[ind+1]
			    		end

			    		for ind in first(ℓ′_range)-ℓ:last(ℓ′_range)-ℓ-1
			    			@. Yℓ′_n2_arr[ind] = Yℓ′_n2_arr[ind+1]
			    		end

			    		Y = Ylmatrix(last(ℓ′_range),x1,n_range=0:0)
		    			Yℓ′_n1_arr[last(ℓ′_range)-ℓ][axes(Y)...] = Y
			    		Y = Ylmatrix(last(ℓ′_range),x2,n_range=0:0)
			    		Yℓ′_n2_arr[last(ℓ′_range)-ℓ][axes(Y)...] = Y
			    	else
			    		# re-initialize the Yℓ′ arrays
			    		for ℓ′ in ℓ′_range
			    			Y = Ylmatrix(ℓ′,x1,n_range=0:0)
			    			Yℓ′_n1_arr[ℓ′-ℓ][axes(Y)...] = Y
			    			Y = Ylmatrix(ℓ′,x2,n_range=0:0)
			    			Yℓ′_n2_arr[ℓ′-ℓ][axes(Y)...] = Y
			    		end
			    	end

			    	ℓ_prev=ℓ

				    for ℓ′ in ℓ′_range

				    	# ignore line-of-sight projection, just the β=γ=0 component
				    	# check if (ℓ′,ℓ) has already been computed
				    	# We can use the fact that Yℓ′ℓs0(n2,n1) = (-1)^(ℓ+ℓ′+s)Yℓℓ′s0(n1,n2)
				    	# If we have computed Yℓℓ′s0(n1,n2) already in the previous step,
				    	# then we can simply read it in from the dictionary

				    	if haskey(Y12,(ℓ′,ℓ))
				    		Yℓ′ℓ_s0_n1n2 = Y12[(ℓ′,ℓ)]
				    	elseif haskey(Y21,(ℓ,ℓ′))
				    		Yℓ′ℓ_s0_n1n2 = Y21[(ℓ,ℓ′)] .* ((-1)^(ℓ+ℓ′+s) for s in axes(Y21[(ℓ,ℓ′)],1))
				    	else
				    		# compute and store it
				    		Yℓ′ℓ_s0_n1n2 = BiPoSH_s0(ℓ′,ℓ,1:s_max,0,0,x1,x2,Y_ℓ₂=Yℓ_n2,
				    						Y_ℓ₁=Yℓ′_n1_arr[ℓ′-ℓ])
				    		Y12[(ℓ′,ℓ)] = Yℓ′ℓ_s0_n1n2
				    	end
				    	
				    	if haskey(Y21,(ℓ′,ℓ))
				    		Yℓ′ℓ_s0_n2n1 = Y21[(ℓ′,ℓ)]
				    	elseif haskey(Y12,(ℓ,ℓ′))
				    		Yℓ′ℓ_s0_n2n1 = Y12[(ℓ,ℓ′)].* ((-1)^(ℓ+ℓ′+s) for s in axes(Y12[(ℓ,ℓ′)],1))
				    	else
				    		Yℓ′ℓ_s0_n2n1 = BiPoSH_s0(ℓ′,ℓ,1:s_max,0,0,x2,x1,Y_ℓ₂=Yℓ_n1,
				    						Y_ℓ₁=Yℓ′_n2_arr[ℓ′-ℓ])
				    		Y21[(ℓ′,ℓ)] = Yℓ′ℓ_s0_n2n1
				    	end

			    		# Compute the CG coefficients that appear in fℓ′ℓsω
			    		for t=-1:0,s in 1:s_max,η=-1:1
			    			if (ℓ′<abs(η)) || (ℓ<abs(η+t))
			    				Cℓ′ℓ[η,s,t] = 0
			    			else
			    				Cℓ′ℓ[η,s,t] = clebschgordan(ℓ′,-η,ℓ,η+t,s,t)
			    			end
			    		end

			    		proc_id_mode_G_x1,ℓ′ω_index_G_x1_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ′,ω_ind),num_procs_x1)
			    		proc_id_mode_G_x2,ℓ′ω_index_G_x2_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ′,ω_ind),num_procs_x2)
						
	    				# Green functions based at the observation point for ℓ′
	    				G = read(Gfn_fits_files_obs1[proc_id_mode_G_x1][1],:,:,1:2,1,1,ℓ′ω_index_G_x1_file)
			    		@. G_x1[:,0] = G[:,1,1] + im*G[:,2,1]
			    		@. G_x1[:,1] = G[:,1,2] + im*G[:,2,2]

			    		if !obs_at_same_height
		    				G = read(Gfn_fits_files_obs2[proc_id_mode_G_x2][1],:,:,1:2,1,1,ℓ′ω_index_G_x2_file)

		    				@. G_x2[:,0] = G[:,1,1] + im*G[:,2,1]
				    		@. G_x2[:,1] = G[:,1,2] + im*G[:,2,2]
				    	end

			    		G = nothing

			    		# precompute the radial term in f, α=0
			    		if 0 in K_components
				    		for η=0:1
				    			@. f_radial_0_r₁[:,η] = (-1)^η * G_x1[:,η] * drGsrc[:,η]
				    			if !obs_at_same_height
				    				@. f_radial_0_r₂[:,η] = (-1)^η * G_x2[:,η] * drGsrc[:,η]
				    			end
				    		end
			    		end

			    		# α=1
			    		if (-1 in K_components) || (1 in K_components)
				    		for η=-1:1
				    			@. f_radial_1_r₁[:,η] = (-1)^η * 1/r * G_x1[:,abs(η)] * 
				    				( Ω(ℓ,η)*Gsrc[:,abs(η)] - ((η != -1) ? Gsrc[:,abs(η-1)] : 0) )
				    			if !obs_at_same_height
				    				@. f_radial_1_r₂[:,η] = (-1)^η * 1/r * G_x2[:,abs(η)] * 
				    				( Ω(ℓ,η)*Gsrc[:,abs(η)] - ((η != -1) ? Gsrc[:,abs(η-1)] : 0) )
				    			end
				    		end
				    	end

			    		for s in LinearIndices(Yℓ′ℓ_s0_n2n1)
			    			# radial component (for all s)
			    			
			    			if 0 in K_components
			    				fℓ′ℓsω_r₁[:,0] .= sum(f_radial_0_r₁[:,abs(η)]*Cℓ′ℓ[η,s,0] for η=-1:1)
			    				if !obs_at_same_height
			    					fℓ′ℓsω_r₂[:,0] .= sum(f_radial_0_r₂[:,abs(η)]*Cℓ′ℓ[η,s,0] for η=-1:1)
			    				else
			    					fℓ′ℓsω_r₂[:,0] .= fℓ′ℓsω_r₁[:,0]
			    				end
			    			end

			    			if (-1 in K_components) || (1 in K_components)
			    				fℓ′ℓsω_r₁[:,1] .= sum(f_radial_1_r₁[:,η]*Cℓ′ℓ[η,s,-1] for η=-1:1)
			    				if !obs_at_same_height
			    					fℓ′ℓsω_r₂[:,1] .= sum(f_radial_1_r₂[:,η]*Cℓ′ℓ[η,s,-1] for η=-1:1)
			    				else
			    					fℓ′ℓsω_r₂[:,1] .= fℓ′ℓsω_r₁[:,1]
			    				end
			    			end

			    			if isodd(ℓ+ℓ′+s) && -1 in K_components
			    				# tangential - component (K⁺ - K⁻), only for odd l+l′+s
								# only imag part calculated, the actual kernel is iK
								# extra factor of 2 from the (1 - (-1)^(ℓ+ℓ′+s)) term
								@. K[:,-1,s] += 2 * dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
					     					2real(conj(h_ω_arr[ω_ind]) *
					     					(conj(fℓ′ℓsω_r₁[:,1])*G_r₂_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
					     						+ conj(G_r₁_rsrc)*fℓ′ℓsω_r₂[:,1]*Yℓ′ℓ_s0_n2n1[s]) )
			    			end

			    			if 0 in K_components
						     	@. K[:,0,s] +=  dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
						     					2real(conj(h_ω_arr[ω_ind]) *
						     					(conj(fℓ′ℓsω_r₁[:,0])*G_r₂_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
						     						+ conj(G_r₁_rsrc)*fℓ′ℓsω_r₂[:,0]*Yℓ′ℓ_s0_n2n1[s]) )
						    end
							
							if iseven(ℓ+ℓ′+s) && 1 in K_components
								# tangential + component (K⁺ + K⁻), only for even l+l′+s
								# extra factor of 2 from the (1 + (-1)^(ℓ+ℓ′+s)) term
								@. K[:,1,s] +=  2 * dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
					     					2real(conj(h_ω_arr[ω_ind]) *
					     					(conj(fℓ′ℓsω_r₁[:,1])*G_r₂_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
					     						+ conj(G_r₁_rsrc)*fℓ′ℓsω_r₂[:,1]*Yℓ′ℓ_s0_n2n1[s]) )
							end
					    end
					end
				end
			end

			@. K *=  2r^2 * ρ

			close.(values(Gfn_fits_files_src))
			close.(values(Gfn_fits_files_obs1))
			close.(values(Gfn_fits_files_obs2))

			if rank_node == 0
				for n in 1:np_node-1
					K .+= take!(channel_on_node)
				end
				put!(channel_on_node,K)
			else
				put!(channel_on_node,K)
			end

			return nothing
		end

		procs_used = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs_used)

		T = OffsetArray{Float64,3,Array{Float64,3}} # type of arrays to be added to the channels
		return pmapsum(T,summodes,procs_used)
	end

	function flow_axisymmetric_srange(n1::Point2D,n2::Point2D,s_max;
		ℓ_range=nothing,ν_ind_range=nothing,bounce_no=1,r_src=Rsun-75e5,r_obs=Rsun-75e5,K_components=-1:1)

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_obs = Gfn_path_from_source_radius(r_obs)

		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs dν ν_start_zeros

		num_procs_obs = load(joinpath(Gfn_path_obs,"parameters.jld2"),"num_procs")

		ℓ_range = intersect_fallback(ℓ_range,ℓ_arr)
		ν_ind_range = intersect_fallback(ν_ind_range,1:Nν_Gfn)
		
		h_ω_arr = h(n1,n2,bounce_no=bounce_no,ℓ_range=ℓ_range,r_src=r_src,r_obs=r_obs).h_ω[ν_start_zeros .+ ν_ind_range] # only in range

		r_obs_ind = argmin(abs.(r .- r_obs))

		dω = dν*2π

		function summodes(rank,rank_node,np_node,channel_on_node)

			ℓ_ωind_iter_on_proc = split_product_across_processors(ℓ_range,ν_ind_range,num_workers,rank)
			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			# Get a list of all modes that will be accessed.
			# This can be used to open the fits files before the loops begin.
			# This will cut down on FITS IO costs

			# Gℓ′ω(r,robs) files
			first_mode = first(ℓ_ωind_iter_on_proc)
			last_mode = last(collect(ℓ_ωind_iter_on_proc))
			ℓ′_min_first_mode = max(minimum(ℓ_arr),abs(first_mode[1]-s_max))
			ℓ′_max_last_mode = min(maximum(ℓ_arr),last_mode[1]+s_max)
			modes_minmax = minmax_from_split_array(ℓ_ωind_iter_on_proc)
			ℓ_max_proc = modes_minmax.ℓ_max
			ℓ′_max_proc =  ℓ_max_proc + s_max
			
			proc_id_min_Gobs = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,
									(ℓ′_min_first_mode,first_mode[2]),num_procs_obs)
			proc_id_max_Gobs = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,
									(ℓ′_max_last_mode,last_mode[2]),num_procs_obs)

			Gfn_fits_files_obs = Gfn_fits_files(Gfn_path_obs,proc_id_min_Gobs:proc_id_max_Gobs)

			K = zeros(nr,minimum(K_components):maximum(K_components),1:s_max)

			Gsrc = zeros(ComplexF64,nr,0:1)
			drGsrc = zeros(ComplexF64,nr,0:1)
			Gobs = zeros(ComplexF64,nr,0:1)

			# temporary array to precompute the radial part, indices are (r,η)
			f_radial_0_robs = zeros(ComplexF64,nr,0:1) # the η=-1 and η=1 terms are identical
			f_radial_1_robs = zeros(ComplexF64,nr,-1:1)
			fℓ′ℓsω_robs = zeros(ComplexF64,nr,0:1)

			# Clebsch Gordan coefficients, indices are (s,η,t)
			Cℓ′ℓ = zeros(-1:1,1:s_max,-1:0)

			# Cache Ylmn arrays to speed up computation of BiPoSH_s0
			# Create an OffsetArray of the Ylmn OffsetArrays
			T = OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
			Yℓ′_n1_arr = OffsetArray{T}(undef,-s_max:s_max)
   			Yℓ′_n2_arr = OffsetArray{T}(undef,-s_max:s_max)
  			# The array indices will be ℓ′-ℓ values to ensure that we can 
  			# overwrite the preallocated arrays
  			  
   			for ind in eachindex(Yℓ′_n1_arr)
   				Yℓ′_n1_arr[ind] = zeros(ComplexF64,-ℓ′_max_proc:ℓ′_max_proc,0:0)
   			end

   			for ind in eachindex(Yℓ′_n2_arr)
   				Yℓ′_n2_arr[ind] = zeros(ComplexF64,-ℓ′_max_proc:ℓ′_max_proc,0:0)
   			end

			# Cache bipolar spherical harmonics in a dict on each worker
			Y12 = Dict{NTuple{2,Int64},OffsetArray{ComplexF64,3,Array{ComplexF64,3}}}()
			Y21 = Dict{NTuple{2,Int64},OffsetArray{ComplexF64,3,Array{ComplexF64,3}}}()

			# keep track of ℓ to cache Yℓ′ by rolling arrays
			# if ℓ changes by 1 arrays can be rolled
			# if the ℓ wraps back then δℓ will be negative. 
			# In this case we need to recompute the Yℓ′ arrays
			ℓ_prev = first_mode[1]

			# Loop over the Greenfn files
			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_src_Gfn_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_src_Gfn_file)

					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_src_Gfn_file,(ℓ,ω_ind))

					ω = dω*(ν_start_zeros + ω_ind)

		 	 		G = read(Gsrc_file[1],:,:,1:2,1,1,ℓω_index_Gsrc_file)
	
		 	 		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		 	 		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		 	 		G_robs_rsrc = Gsrc[r_obs_ind,0]
	
		 	 		if 0 in K_components
		 	 			G = read(Gsrc_file[1],:,:,1:2,1,2,ℓω_index_Gsrc_file)
			    		@. drGsrc[:,0] = G[:,1,1] + im*G[:,2,1]
			    		@. drGsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		 	   		end
	
		 	   		G = nothing

		 	   		ℓ′_range = intersect(ℓ_arr,abs(ℓ-s_max):ℓ+s_max)
	
		 	   		# Precompute Ylmatrix to speed up evaluation of BiPoSH_s0
		 	   		Yℓ_n2 = Ylmatrix(ℓ,n2,n_range=0:0)
		 	   		Yℓ_n1 = Ylmatrix(ℓ,n1,n_range=0:0)    
	
		 	   		if (ℓ - ℓ_prev) == 1
			    		# roll the Yℓ′ arrays
			    		for ind in first(ℓ′_range)-ℓ:last(ℓ′_range)-ℓ-1
			    			@. Yℓ′_n1_arr[ind] = Yℓ′_n1_arr[ind+1]
			    		end

			    		for ind in first(ℓ′_range)-ℓ:last(ℓ′_range)-ℓ-1
			    			@. Yℓ′_n2_arr[ind] = Yℓ′_n2_arr[ind+1]
			    		end

			    		Y = Ylmatrix(last(ℓ′_range),n1,n_range=0:0)
		 	   			Yℓ′_n1_arr[last(ℓ′_range)-ℓ][axes(Y)...] = Y
			    		Y = Ylmatrix(last(ℓ′_range),n2,n_range=0:0)
			    		Yℓ′_n2_arr[last(ℓ′_range)-ℓ][axes(Y)...] = Y
			    	else
			    		# re-initialize the Yℓ′ arrays
			    		for ℓ′ in ℓ′_range
			    			Y = Ylmatrix(ℓ′,n1,n_range=0:0)
			    			Yℓ′_n1_arr[ℓ′-ℓ][axes(Y)...] = Y
			    			Y = Ylmatrix(ℓ′,n2,n_range=0:0)
			    			Yℓ′_n2_arr[ℓ′-ℓ][axes(Y)...] = Y
			    		end
			    	end

			    	ℓ_prev=ℓ

				    for ℓ′ in ℓ′_range

				    	# ignore line-of-sight projection, just the β=γ=0 component
				    	# check if (ℓ′,ℓ) has already been computed
				    	# We can use the fact that Yℓ′ℓs0(n2,n1) = (-1)^(ℓ+ℓ′+s)Yℓℓ′s0(n1,n2)
				    	# If we have computed Yℓℓ′s0(n1,n2) already in the previous step,
				    	# then we can simply read it in from the dictionary

				    	if haskey(Y12,(ℓ′,ℓ))
				    		Yℓ′ℓ_s0_n1n2 = Y12[(ℓ′,ℓ)]
				    	elseif haskey(Y21,(ℓ,ℓ′))
				    		Yℓ′ℓ_s0_n1n2 = Y21[(ℓ,ℓ′)] .* ((-1)^(ℓ+ℓ′+s) for s in axes(Y21[(ℓ,ℓ′)],1))
				    	else
				    		# compute and store it
				    		Yℓ′ℓ_s0_n1n2 = BiPoSH_s0(ℓ′,ℓ,1:s_max,0,0,n1,n2,Y_ℓ₂=Yℓ_n2,
				    						Y_ℓ₁=Yℓ′_n1_arr[ℓ′-ℓ])
				    		Y12[(ℓ′,ℓ)] = Yℓ′ℓ_s0_n1n2
				    	end
				    	
				    	if haskey(Y21,(ℓ′,ℓ))
				    		Yℓ′ℓ_s0_n2n1 = Y21[(ℓ′,ℓ)]
				    	elseif haskey(Y12,(ℓ,ℓ′))
				    		Yℓ′ℓ_s0_n2n1 = Y12[(ℓ,ℓ′)].* ((-1)^(ℓ+ℓ′+s) for s in axes(Y12[(ℓ,ℓ′)],1))
				    	else
				    		Yℓ′ℓ_s0_n2n1 = BiPoSH_s0(ℓ′,ℓ,1:s_max,0,0,n2,n1,Y_ℓ₂=Yℓ_n1,
				    						Y_ℓ₁=Yℓ′_n2_arr[ℓ′-ℓ])
				    		Y21[(ℓ′,ℓ)] = Yℓ′ℓ_s0_n2n1
				    	end

			    		# Compute the CG coefficients that appear in fℓ′ℓsω
			    		for t=-1:0,s in 1:s_max,η=-1:1
			    			if (ℓ′<abs(η)) || (ℓ<abs(η+t))
			    				Cℓ′ℓ[η,s,t] = 0
			    			else
			    				Cℓ′ℓ[η,s,t] = clebschgordan(ℓ′,-η,ℓ,η+t,s,t)
			    			end
			    		end

			    		proc_id_mode_Gobs,ℓ′ω_index_Gobs_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ′,ω_ind),num_procs_obs)
			    		
			    		# fits_init_time += @elapsed begin
			    		# # Gobs_file = FITS(joinpath(Gfn_path_obs,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs),"r")
			    		# Gobs_file = Gobs_files[proc_id_mode_Gobs]
			    		# end
						
		  		  		# Green functions based at the observation point for ℓ′
		  		  		G = read(Gfn_fits_files_obs[proc_id_mode_Gobs][1],:,:,1:2,1,1,ℓ′ω_index_Gobs_file)
			    		@. Gobs[:,0] = G[:,1,1] + im*G[:,2,1]
			    		@. Gobs[:,1] = G[:,1,2] + im*G[:,2,2]

			    		# fits_init_time += @elapsed close(Gobs_file)

			    		G = nothing

			    		# precompute the radial term in f, α=0
			    		if 0 in K_components
				    		for η=0:1
				    			@. f_radial_0_robs[:,η] = (-1)^η * Gobs[:,η] * drGsrc[:,η]
				    		end
			    		end

			    		# α=1
			    		if (-1 in K_components) || (1 in K_components)
				    		for η=-1:1
				    			@. f_radial_1_robs[:,η] = (-1)^η * 1/r * Gobs[:,abs(η)] * 
				    				( Ω(ℓ,η)*Gsrc[:,abs(η)] - ((η != -1) ? Gsrc[:,abs(η-1)] : 0) )
				    		end
				    	end

			    		for s in LinearIndices(Yℓ′ℓ_s0_n2n1)
			    			# radial component (for all s)
			    			
			    			if 0 in K_components
			    				fℓ′ℓsω_robs[:,0] .= sum(f_radial_0_robs[:,abs(η)]*Cℓ′ℓ[η,s,0] for η=-1:1) 
			    			end

			    			if (-1 in K_components) || (1 in K_components)
			    				fℓ′ℓsω_robs[:,1] .= sum(f_radial_1_robs[:,η]*Cℓ′ℓ[η,s,-1] for η=-1:1)
			    			end

			    			if isodd(ℓ+ℓ′+s) && -1 in K_components
			    				# tangential - component (K⁺ - K⁻), only for odd l+l′+s
								# only imag part calculated, the actual kernel is iK
								# extra factor of 2 from the (1 - (-1)^(ℓ+ℓ′+s)) term
								@. K[:,-1,s] += 2 * dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
					     					2real(conj(h_ω_arr[ω_ind]) *
					     					(conj(fℓ′ℓsω_robs[:,1])*G_robs_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
					     						+ conj(G_robs_rsrc)*fℓ′ℓsω_robs[:,1]*Yℓ′ℓ_s0_n2n1[s]) )
			    			end

			    			if 0 in K_components
						     	@. K[:,0,s] +=  dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
						     					2real(conj(h_ω_arr[ω_ind]) *
						     					(conj(fℓ′ℓsω_robs[:,0])*G_robs_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
						     						+ conj(G_robs_rsrc)*fℓ′ℓsω_robs[:,0]*Yℓ′ℓ_s0_n2n1[s]) )
						    end
							
							if iseven(ℓ+ℓ′+s) && 1 in K_components
								# tangential + component (K⁺ + K⁻), only for even l+l′+s
								# extra factor of 2 from the (1 + (-1)^(ℓ+ℓ′+s)) term
								@. K[:,1,s] +=  2 * dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
					     					2real(conj(h_ω_arr[ω_ind]) *
					     					(conj(fℓ′ℓsω_robs[:,1])*G_robs_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
					     						+ conj(G_robs_rsrc)*fℓ′ℓsω_robs[:,1]*Yℓ′ℓ_s0_n2n1[s]) )
							end
					    end
					end
				end
			end

			@. K *= 2r^2 * ρ

			close.(values(Gfn_fits_files_obs))
			close.(values(Gfn_fits_files_src))

			if rank_node == 0
				for n in 1:np_node-1
					K .+= take!(channel_on_node)
				end
				put!(channel_on_node,K)
			else
				put!(channel_on_node,K)
			end

			return nothing
		end
		
		procs_used = workers_active(ℓ_range,ν_ind_range)
		num_workers = length(procs_used)

		T = OffsetArray{Float64,3,Array{Float64,3}} # type of arrays to be added to the channels
		return pmapsum(T,summodes,procs_used)
	end

	function meridional_flow_ψ_srange(x1::SphericalPoint,x2::SphericalPoint,s_max;kwargs...)
		Kv = flow_kernels_srange_t0(x1,x2,s_max;K_components=0:1,kwargs...)

		Kψ_imag = zeros(nr,1:s_max)

		# negative of the imaginary part of the stream function kernel
		@inbounds for s in axes(Kψ_imag,2)
			Kψ_imag[:,s] .= ddr*(Kv[:,1,s]./ρ) .+ @. Kv[:,1,s]/(ρ*r) - 2*Ω(s,0)*Kv[:,0,s]/(ρ*r)
		end

		Kψ_imag .*= -1 # there's an overall minus sign that was not included in the expression above

		# Imaginary part of kernel, actual kernel is im*K
		return Kψ_imag
	end
end

# module kernel3D
# 	using Main.kernel
# 	import WignerD: Ylmatrix

# 	function axisymmetric_flow_kernels_rθ_slice(x1::SphericalPoint,x2::SphericalPoint,s_max;kwargs...)
# 		Kjlm_r = flow_kernels_srange(x1,x2,s_max;kwargs...)
# 		θ_arr = LinRange(0,π,nθ)
# 		# Slice at ϕ=0
# 		n_arr = [Point2D(θ,0) for θ in θ_arr]

# 		K2D_r_θ = zeros(size(Kjlm_r,1))

# 		# The 3D kernel is given by ∑ₛₙ Ks0n(r)* Ys0n(θ,ϕ)


# 	end
# end