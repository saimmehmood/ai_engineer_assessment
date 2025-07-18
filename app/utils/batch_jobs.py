from flask import current_app


def create_job(job_definition, job_queue, job_name, command, job_type, min_memory=None, group_id=None):
    batch_job = {
        'job_definition': job_definition,
        'job_queue': job_queue,
        'job_name': job_name,
        'command': command,
        'env': [{'name': k, 'value': v} for k, v in current_app.config.items()],
        'job_type': job_type,
    }

    if min_memory:
        batch_job['min_memory'] = min_memory

    if group_id:
        batch_job['group_id'] = group_id

    return batch_job


def create_batch(iterable, batch_size):
    """
    Creates a list of batches from an iterable
    :param iterable: iterable to create batches from
    :param batch_size: size of each batch
    :return: list of batches
    """
    return [iterable[i:i + batch_size] for i in range(0, len(iterable), batch_size)]\
        if len(iterable) > batch_size else [iterable]
