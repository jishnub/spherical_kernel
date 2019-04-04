module ParallelUtilities

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

	@assert(proc_id<=num_procs,"processor rank has to be less than number of workers engaged")
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

workers_active(arr) = [p for (rank,p) in enumerate(workers()) 
							if !isempty(split_across_processors(arr,nworkers(),rank))]

workers_active(arr₁,arr₂) = [p for (rank,p) in enumerate(workers()) 
							if !isempty(split_product_across_processors(arr₁,arr₂,nworkers(),rank))]

function minmax_from_split_array(ℓ_ωind_iter)
	ℓ_min,ω_ind_min = first(ℓ_ωind_iter)
	ℓ_max,ω_ind_max = ℓ_min,ω_ind_min
	for (ℓ,ω_ind) in ℓ_ωind_iter
		ℓ_min = min(ℓ_min,ℓ)
		ℓ_max = max(ℓ_max,ℓ)
		ω_ind_min = min(ω_ind_min,ω_ind)
		ω_ind_max = max(ω_ind_max,ω_ind)
	end
	return (ℓ_min=ℓ_min,ℓ_max=ℓ_max,ω_ind_min=ω_ind_min,ω_ind_max=ω_ind_max)
end

function node_remotechannels(::Type{T},procs_used) where {T}
	hostnames = pmap(p->remotecall_fetch(Base.Libc.gethostname,p),procs_used)
	nodes = unique(hostnames);	num_nodes = length(nodes)
	num_procs_node = Dict(node=>count(x->x==node,hostnames) for node in nodes)
	rank_counter = Dict(node=>0 for node in nodes)
	rank_on_node = zeros(Int64,length(procs_used))

	for (ind,host) in enumerate(hostnames)
		rank_on_node[ind] = rank_counter[host]
		rank_counter[host] += 1
	end

	# Create one local channel on each node to carry out reduction,
	node_channels = Dict(
		hostnames[findfirst(x->x==p,procs_used)]=>
		RemoteChannel(
			()->Channel{T}(
				num_procs_node[hostnames[findfirst(x->x==p,procs_used)]]-1),p)
		for p in procs_used[rank_on_node .== 0])

	return rank_on_node,hostnames,num_procs_node,node_channels
end

function pmapsum(::Type{T},f,iterable,args...;kwargs...) where {T}

	procs_used = workers_active(iterable)
	num_workers = length(procs_used)
	rank_on_node,hostnames,num_procs_node,node_channels = node_remotechannels(T,procs_used)

	@sync for (rank,(p,hostname)) in enumerate(zip(procs_used,hostnames))
		@async begin
			iterable_on_proc = split_across_processors(iterable,num_workers,rank)
			remotecall_wait(apply_sum,p,f,iterable_on_proc,args...;
			rank_node=rank_on_node[rank],
			np_node=num_procs_node[hostname],
			channel_on_node=node_channels[hostname],kwargs...)
		end
	end

	# worker at which final reduction takes place
	p = procs_used[findfirst(x->x==0,rank_on_node)]

	K = remotecall_fetch(x->sum(take!.(values(x))),p,node_channels)
	
	finalize.(values(node_channels))
	return K
end

pmapsum(f,procs_used) = pmapsum(Any,f,procs_used)

function pmap_onebatch_per_worker(f,iterable,args...;kwargs...)

	procs_used = workers_active(iterable)
	num_workers = length(procs_used)

	@sync for (rank,p) in enumerate(procs_used)
		@async begin
			iterable_on_proc = split_across_processors(iterable,num_workers,rank)
			@spawnat p f(iterable_on_proc,args...;kwargs...)
		end
	end
	return nothing
end

function sum_at_node!(::Type{T},var::T,rank_node,np_node,
	channel_on_node::RemoteChannel{Channel{T}}) where {T}

	if iszero(rank_node)
		for n in 1:np_node-1
			var += take!(channel_on_node)
		end
		put!(channel_on_node,var)
	else
		put!(channel_on_node,var)
	end
	finalize(channel_on_node)
end

function sum_at_node!(var::T,rank_node,np_node,channel_on_node::RemoteChannel{Channel{T}}) where {T}
	sum_at_node!(T,var,rank_node,np_node,channel_on_node)
end

function apply_sum(f,args...;rank_node,np_node,channel_on_node,kwargs...)
	var = f(args...;kwargs...)
	sum_at_node!(var,rank_node,np_node,channel_on_node)
	return nothing
end

export split_product_across_processors,get_processor_id_from_split_array
export get_processor_range_from_split_array,workers_active,worker_rank
export get_index_in_split_array,procid_and_mode_index,minmax_from_split_array
export node_remotechannels,pmapsum,sum_at_node!,pmap_onebatch_per_worker

end # module
