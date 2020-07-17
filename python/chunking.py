# TODO(eric.cousineau): Is there a super small lib that does this?


def split_into_chunks(x, num_chunks):
    count = len(x)
    num_chunks = min(count, num_chunks)
    chunk_size = count // num_chunks
    index = 0
    chunks = []
    for i in range(num_chunks):
        chunks.append(x[index:index + chunk_size])
        index += chunk_size
    assert len(chunks) == num_chunks
    return chunks, num_chunks


def split_item_list(item_list, item_cost_func, num_chunks):
    """Split @p item_list into @p num_chunks such that all chunks have similar
    sum item cost (as computed by @p item_cost_func)."""
    # TODO(eric.cousineau): Sort items by cost first, then for each chunk, add
    # first item, then iterate until we exhaust the cost?
    if num_chunks == 1:
        return [item_list]
    item_costs = [item_cost_func(x) for x in item_list]
    print(item_costs)
    cost_total = sum(item_costs)
    chunk_cost_target = cost_total / num_chunks
    chunk_list = []
    chunk = []
    chunk_cost = 0
    for item, item_cost in zip(item_list, item_costs):
        past_cost_target = (chunk_cost >= chunk_cost_target)
        need_more_chunks = (len(chunk_list) + 1 < num_chunks)
        if past_cost_target and need_more_chunks:
            # Store current chunk (with previous item(s)), and start a new
            # chunk.
            chunk_list.append(chunk)
            chunk = []
            chunk_cost = 0
        chunk_cost += item_cost
        chunk.append(item)
    # Append final chunk.
    chunk_list.append(chunk)
    assert len(chunk_list) <= num_chunks, (len(chunk_list), num_chunks)
    return chunk_list


def flatten_suite(suite):
    tests = []
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            tests += flatten_suite(test)
        else:
            tests += [test]
    return tests
