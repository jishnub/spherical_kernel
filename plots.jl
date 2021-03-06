using PyPlot,PointsOnASphere,LaTeXStrings,LinearAlgebra,FITSIO,JLD2,FFTW
using PyCall,DelimitedFiles,OffsetArrays,DataFrames,CSV,Printf,ProgressMeter
@pyimport matplotlib.ticker as ticker

SCRATCH = ENV["SCRATCH"]

const Rsun = Main.crosscov.Rsun

function plot_traveltimes_validation(ϕ_low=40,ϕ_high=75;nϕ=5,bounce_no=1,kwargs...)
	
	r_src = get(kwargs,:r_src,Rsun-75e5) :: Float64
	Gfn_path_src = Main.crosscov.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") Nt dt dν Nν

	ϕ2_deg = collect(LinRange(ϕ_low,ϕ_high,nϕ))
	ϕ2_arr = ϕ2_deg.*π/180
	n1 = Point2D(π/2,0)
	n2_arr = [Point2D(π/2,ϕ2) for ϕ2 in ϕ2_arr]

	δτ_FB = zeros(length(n2_arr))
	δτ_rot2 = zeros(length(n2_arr))

	t_inds = Vector{UnitRange}(undef,length(n2_arr))

	Cω_arr = zeros(ComplexF64,Nν,length(n2_arr))
	∂ϕ₂Cω_arr = zeros(ComplexF64,Nν,length(n2_arr))
	hω_arr = zeros(ComplexF64,Nν,length(n2_arr))
	@showprogress 1 "Computing time window " for (ind,n2) in enumerate(n2_arr)
		Cω,∂ϕ₂Cω = Main.crosscov.Cω_∂ϕ₂Cω(n1,n2;kwargs...)
		Cω_arr[:,ind] = Cω
		∂ϕ₂Cω_arr[:,ind] = ∂ϕ₂Cω
		Ct = Main.crosscov.Ct(Cω,dν)
		∂ϕ₂Ct = Main.crosscov.∂ϕ₂Ct(∂ϕ₂Cω,dν)
		t_inds[ind] = Main.crosscov.time_window_indices_by_fitting_bounce_peak(Ct,n1,n2,
								dt=dt,Nt=Nt,bounce_no=bounce_no)
		hω_arr[:,ind] = Main.crosscov.hω(Cω,n1,n2;τ_ind_arr=t_inds[ind],kwargs...)
	end

	println("Computing rotated-frame travel times")
	δτ_rot2 .= Main.traveltimes.δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(
								n1,n2_arr;τ_ind_arr=t_inds,hω=hω_arr,
								Cω=Cω_arr,∂ϕ₂Cω=∂ϕ₂Cω_arr,kwargs...)
	println("Finished computing rotated-frame travel times")

	println("Computing first Born travel times")
	@time δτ_FB .= Main.traveltimes.δτ_uniform_rotation_firstborn_int_K_u(n1,n2_arr;
									hω_arr=hω_arr,kwargs...)
	println("Finished computing first Born travel times")

	percentage_diff = (δτ_rot2-δτ_FB)./δτ_rot2.*100

	ϕ2_arr .*= 180/π # convert to degrees

	ax1 = subplot2grid((3,1),(0,0),rowspan=2)
	plot(ϕ2_arr,δτ_rot2,label="Rotated frame",color="black",ls="dashed")
	plot(ϕ2_arr,δτ_FB,"^",label="First Born",ms=6,ls="None",color="black")
	ylabel("travel time shift [sec]",fontsize=12)
	legend(loc="best")
	ax1[:xaxis][:set_major_formatter](ticker.NullFormatter())
	ax1[:xaxis][:set_major_locator](ticker.MaxNLocator(5))

	ax2 = subplot2grid((3,1),(2,0))
	plot(ϕ2_arr,percentage_diff,"o-",ms=4,zorder=2,color="black")
	ylabel("Difference",fontsize=12)
	xlabel("Angular separation [degrees]",fontsize=12)
	ax2[:axhline](0,ls="dotted",color="black",zorder=0)
	ax2[:margins](y=0.2)
	ax2[:set_yticklabels]([(@sprintf "%.2f" x)*"%" for x in ax2[:get_yticks]()])
	ax2[:xaxis][:set_major_locator](ticker.MaxNLocator(5))
	ax2[:yaxis][:set_major_locator](ticker.MaxNLocator(3))

	tight_layout()
	gcf()[:subplots_adjust](hspace=0)

	dτ_arr = DataFrame(dist=ϕ2_deg,t_inds=t_inds,dt_FB=δτ_FB,dt_rot=δτ_rot2,
		percentage_diff=percentage_diff)

	savefig("$SCRATCH/traveltimes_validation.png")
	CSV.write("$SCRATCH/travel_time_shifts_uniform_rotation",dτ_arr,delim=' ')

	dτ_arr
end

function compute_K10_different_arrival_bounces()
	n1 = Point2D(π/2,0); n2 = Point2D(π/2,π/3)
	kernels = zeros(Main.kernel.nr,3)

	for (ind,bounce_no) in enumerate((1,2,4))
		kernels[:,ind] .= imag.(Main.kernel.kernel_uniform_rotation_uplus(n1,n2,ℓ_range=20:100,bounce_no=bounce_no))
	end

	mkpath("$SCRATCH/kernels")
	f = FITS("$SCRATCH/kernels/K10_different_bounces.fits","w")
	write(f,kernels)
	close(f)

	return kernels
end

function plot_K10_different_arrival_bounces()

	kernel_file = "$SCRATCH/kernels/K10_different_bounces.fits"
	if isfile(kernel_file)
		f = FITS(kernel_file)
		kernels = read(f[1])
		close(f)
	else
		kernels = compute_K10_different_arrival_bounces()
	end

	r = copy(Main.kernel.r)
	Rsun = Main.kernel.Rsun
	r ./= Rsun
	subplot(211)
	plot(r,normalize(kernels[:,1],Inf),label="first bounce")
	xlim(0.8,r[end])
	legend(loc="best")
	title(L"Normalized $K_{10}(r)$",fontsize=12)

	subplot(212)
	plot(r,kernels[:,3],label="fourth bounce")
	xlim(0.8,r[end])
	legend(loc="best")
	
	xlabel(L"$r/R_\odot$",fontsize=12)
	

	tight_layout()
end

function plot_Ks0()
	kernel_file = "$SCRATCH/kernels/Ks0_lmax100.fits"
	
	f = FITS(kernel_file)
	kernel = read(f[1])
	s_max = size(kernel,3)
	kernel = OffsetArray(kernel,axes(kernel,1),-1:1,axes(kernel,3))
	close(f)

	r = copy(Main.kernel.r)
	c = copy(Main.kernel.c)
	Rsun = Main.kernel.Rsun
	r ./= Rsun


	ax1 = subplot(311)
	title("Kernel components for 60° separation")
	for s=1:4:s_max
		plot(r,kernel[:,0,s],label="s=$s")
	end
	ylabel("Kr_s0")
	legend(loc="best")
	xlim(0.9,r[end])
	ax1[:xaxis][:set_major_formatter](ticker.NullFormatter())

	ax2 =subplot(312)	
	for s=1:4:s_max
		plot(r,kernel[:,1,s],label="s=$s")
	end
	legend(loc="best")
	xlim(0.9,r[end])
	ylabel("Kθ_s0")
	xlabel(L"$r/R_\odot$",fontsize=12)
	ax2[:xaxis][:set_major_formatter](ticker.NullFormatter())

	ax3 =subplot(313)
	for s=1:4:s_max
		plot(r,kernel[:,-1,s],label="s=$s")
	end
	ylabel("Kϕ_s0")
	legend(loc="best")
	xlim(0.9,r[end])
	xlabel(L"$r/R_\odot$",fontsize=12)

	tight_layout()
end

function plot_time_distance(;kwargs...) # time in hours

	r_src = get(kwargs,:r_src,Main.crosscov.r_src_default) :: Float64
	t_max = get(kwargs,:t_max,6)
	t_min = get(kwargs,:t_min,0.3)
	ℓ_range = get(kwargs,:ℓ_range,20:100)
	nϕ = get(kwargs,:nϕ,maximum(ℓ_range))

	Gfn_path_src = Main.Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt Nt

	t_max_ind = Int(floor(t_max*60^2/dt))+1
	t_min_ind = Int(floor(t_min*60^2/dt))+1
	t_inds = t_min_ind:t_max_ind

	t = t_inds.*dt./60^2

	ϕ = LinRange(5,90,nϕ).*π/180

	C = Main.crosscov.CΔϕt(τ_ind_arr=t_inds,ℓ_range=ℓ_range,Δϕ_arr=ϕ)
	C = copy(transpose(C))
	C60 = Main.crosscov.Ct(π/3,ℓ_range=ℓ_range)[t_inds]

	ax1 = subplot2grid((1,3),(0,0),colspan=2)
	ax2 = subplot2grid((1,3),(0,2),colspan=1)
	ax1[:pcolormesh](ϕ.*180/π,t,C./maximum(abs,C),
		cmap="Greys",vmax=1,vmin=-1,rasterized=true)
	ax1[:set_xlabel]("Angular separation [degrees]",fontsize=12);
	ax1[:set_ylabel]("Time [hours]",fontsize=12);
	ax1[:set_title]("Time-distance diagram",fontsize=12);
	ax1[:axvline](60,color="black",ls="dashed",lw=0.6)

	ax2[:plot](C60,t,lw=0.7)
	ax2[:yaxis][:set_major_formatter](ticker.NullFormatter())
	ax2[:set_ylim](ax1[:get_ylim]())

	tight_layout()
end

function plot_time_distance_bounce_filtered(;bounce_no=1,kwargs...)
	r_src = get(kwargs,:r_src,Main.crosscov.r_src_default) :: Float64
	t_max = get(kwargs,:t_max,6)
	t_min = get(kwargs,:t_min,0.3)
	ℓ_range = get(kwargs,:ℓ_range,20:100)
	nϕ = get(kwargs,:nϕ,maximum(ℓ_range))
	
	Gfn_path_src = Main.Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt Nt

	t_max_ind = Int(floor(t_max*60^2/dt))+1
	t_min_ind = Int(floor(t_min*60^2/dt))+1
	t_inds = t_min_ind:t_max_ind

	t = t_inds.*dt./60^2

	ϕ = LinRange(5,90,nϕ).*π/180

	C = Main.crosscov.CΔϕt(τ_ind_arr=t_inds,ℓ_range=ℓ_range,Δϕ_arr=ϕ)
	C = copy(transpose(C))

	C_filt = copy(C)
	f_t = zeros(size(C_filt))
	for (ind,Δϕ_i) in enumerate(ϕ)
		τ_low_ind,τ_high_ind = Main.crosscov.time_window_bounce_filter(Δϕ_i,dt,bounce_no)
		f_t[τ_low_ind:τ_high_ind,ind] .= 1
	end

	C_filt .*= f_t

	ax1 = subplot(121)
	ax2 = subplot(122)

	ax1[:pcolormesh](ϕ.*180/π,t,C./maximum(abs,C),
		cmap="Greys",vmax=1,vmin=-1,rasterized=true)
	ax1[:set_xlabel]("Angular separation [degrees]",fontsize=12);
	ax1[:set_ylabel]("Time [hours]",fontsize=12);
	ax1[:set_title]("Time-distance diagram",fontsize=12);

	ax2[:pcolormesh](ϕ.*180/π,t,C_filt./maximum(abs,C_filt),
		cmap="Greys",vmax=1,vmin=-1,rasterized=true)
	ax2[:set_xlabel]("Angular separation [degrees]",fontsize=12);
	ax2[:set_ylabel]("Time [hours]",fontsize=12);
	ax2[:set_title]("Filtered bounce",fontsize=12);

end

function plot_C_spectrum(;ℓ_range=nothing,ν_ind_range=nothing)

	r_src=Main.crosscov.Rsun-75e5
	Gfn_path_src = Main.Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr

	Nν_Gfn = length(ν_arr)

	Cωℓ = Main.crosscov.Cωℓ_spectrum(ℓ_range=ℓ_range,ν_ind_range=ν_ind_range)
	ν_ind_range,ℓ_range = axes(Cωℓ)
	ℓ_edges = collect(first(ℓ_range):last(ℓ_range)+1)
	ν_edges = collect(ν_arr[first(ν_ind_range):last(ν_ind_range)]).*1e3
	spec = Cωℓ.parent./maximum(abs,Cωℓ)

	pcolormesh(ℓ_edges,ν_edges,spec,
		cmap="Oranges",vmax=0.5,vmin=0,rasterized=true);
	xlabel(L"$\ell$",fontsize=12);
	ylabel("Frequency [mHz]",fontsize=12);
	title("Spectrum of cross covariance",fontsize=12);
	tight_layout()
	savefig("C_spectrum.eps")
end

function plot_Ct_groups_of_modes(;t_max=6,t_min=0.3)
	r_src=Main.crosscov.Rsun-75e5
	Gfn_path_src = Main.Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt Nt

	t_max_ind = Int(floor(t_max*60^2/dt))+1
	t_min_ind = Int(floor(t_min*60^2/dt))+1

	t_inds = t_min_ind:t_max_ind
	t = t_inds.*dt/60^2
	nt = length(t)

	n1 = Point2D(π/2,0)
	n2 = Point2D(π/2,π/3)

	ℓ_ranges = Base.Iterators.partition(20:99,20)

	C = zeros(nt,length(ℓ_ranges)+1)

	for (ind,ℓ_range) in enumerate(ℓ_ranges)
		C[:,ind] = Main.crosscov.Ct(n1,n2,ℓ_range=ℓ_range)[t_inds]
	end

	C[:,end] = Main.crosscov.Ct(n1,n2,ℓ_range=20:99)[t_inds]

	fig,ax = subplots(nrows=size(C,2),ncols=1)
	for axis in ax
		axis[:set_xlim](first(t),last(t))
	end

	ax[1][:set_title]("Cross covariance (Δϕ=60 degrees)")

	for (ind,ℓ_range) in enumerate(ℓ_ranges)
		ax[ind][:plot](t,C[:,ind],label="ℓ=$(ℓ_range[1]:ℓ_range[end])")
		ax[ind][:xaxis][:set_major_formatter](ticker.NullFormatter())
		ax[ind][:yaxis][:set_major_locator](ticker.MaxNLocator(3))
		ax[ind][:legend](loc="upper right",bbox_to_anchor=(1.01,1.5))
	end

	ax[end][:plot](t,C[:,end])
	ax[end][:set_title]("Sum over ℓ")
	ax[end][:yaxis][:set_major_locator](ticker.MaxNLocator(3))
	
	xlabel("Time [hours]",fontsize=12)
	tight_layout()
	savefig("Ct_lranges.eps")
end

function plot_kernel_timing_scaling_benchmark(n1=Point2D(π/2,0),n2=Point2D(π/2,π/3);
	s_max=10,ℓ_range=20:30:100)

	ns = 3; chunksize = max(1,div(s_max-1,ns))
	s_range = 1:chunksize:s_max
	evaltime = zeros(length(ℓ_range),length(s_range))
	p = Progress(length(evaltime), 1) 
	for (s_ind,s_max) in enumerate(s_range), (ℓind,ℓ) in enumerate(ℓ_range)
		evaltime[ℓind,s_ind] = @elapsed Main.kernel.flow_axisymmetric_without_los(
									n1,n2,s_max,ℓ_range=20:ℓ);
		next!(p)
	end

	subplot(211)
	for (s_ind,s_max) in enumerate(s_range)
		plot(ℓ_range,evaltime[:,s_ind],label="s_max = $s_max",marker="o",ms=4,ls="solid")
	end
	xlabel("Maximum ℓ",fontsize=12)
	ylabel("Runtime [sec]",fontsize=12)
	legend(loc="best",fontsize=12)

	subplot(212)
	plot(s_range,evaltime[end,:],marker="o",ms=4,ls="solid",label="ℓ = $(last(ℓ_range))")
	xlabel("Maximum s",fontsize=12)
	ylabel("Runtime [sec]",fontsize=12)

	tight_layout()
end

function plot_h(x1,x2;kwargs...)
	
	r_src = get(kwargs,:r_src,Rsun-75e5)
	Gfn_path_src = Main.crosscov.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_start_zeros ν_arr Nt dt dν Nν_Gfn

	Cω_x1x2 = Main.crosscov.Cω(x1,x2;kwargs...)
	C_t = Main.crosscov.@fft_ω_to_t(Cω_x1x2)
	τ_ind_arr = Main.crosscov.time_window_indices_by_fitting_bounce_peak(C_t,x1,x2;dt=dt,kwargs...)
	ht = Main.crosscov.ht(Cω_x1x2,x1,x2;τ_ind_arr=τ_ind_arr,kwargs...)
	hω = Main.crosscov.@fft_t_to_ω(ht)

	subplot(411)
	plot(ν_arr,real(Cω_x1x2[ν_start_zeros .+ (1:Nν_Gfn)]))
	title("C(x₁,x₂,ω)")

	ax2=subplot(412)
	plot((1:Nt).*dt,C_t,color="black")
	axvline(τ_ind_arr[1]*dt,ls="solid")
	axvline(τ_ind_arr[end]*dt,ls="solid")
	title("C(x₁,x₂,t)")

	subplot(413,sharex=ax2)
	plot((1:Nt).*dt,ht,color="black")
	xlim(0,60^2*6)

	title("h(x₁,x₂,t)")

	subplot(414)
	plot(ν_arr,imag(hω[ν_start_zeros .+ (1:Nν_Gfn)]),label="imag")
	plot(ν_arr,real(hω[ν_start_zeros .+ (1:Nν_Gfn)]),label="real")
	legend(loc="best")
	title("h(x₁,x₂,ω)")

	tight_layout()
end

function plot_kernel_latitudinal_slice(θ₁,θ₂,s_max=10;kwargs...)
	n1 = Point2D(θ₁,0)
	n2 = Point2D(θ₂,0)

	nθ = 100
	θ = LinRange(0,π,nθ)

	Kr_rθ = Kr_longitudinal_slice(n1,n2,s_max;K_components=0:0,nθ=nθ)

	r = copy(Main.kernel.r)
	c = copy(Main.kernel.c)
	Rsun = Main.kernel.Rsun
	r ./= Rsun

	subplot(131)
	plot(r,θ,Kr_rθ)
end