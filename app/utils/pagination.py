def format_results(
    response: dict,
    page: int,
    per_page: int,
) -> dict:
    """
    Formats the results of the search query (Elasticsearch/Opensearch response)
    :param response: Elasticsearch/Opensearch response
    :param page: Page number
    :param per_page: Number of items per page
    :return:
    """
    hits = response.get('hits', {}).get('hits', [])
    total = response.get('hits', {}).get('total', {}).get('value', 0)

    return {
        'items': [hit['_source'] for hit in hits],
        **generate_pagination_metadata(
            page=page,
            per_page=per_page,
            total=total
        )
    }


def generate_pagination_metadata(
    page: int,
    per_page: int,
    total: int
) -> dict:
    """
    Generates pagination metadata
    :param page:
    :param per_page:
    :param total:
    :return:
    """

    max_pages = min(total // per_page + 1, 1000)
    pages = min(total // per_page, 10000 // per_page)

    if total % per_page > 0:
        pages += 1

    return {
        'page': str(page),
        'per_page': str(per_page),
        'has_next': page < max_pages,
        'has_prev': page > 1,
        'next_num': str(min(page + 1, max_pages)) if page < max_pages else None,
        'prev_num': str(page - 1) if page > 1 else None,
        'pages': str(pages),
        'total': str(min(total, 10000)),
    }