import re
from collections import defaultdict


def parse_entity_label(entity_label):
    """This function parses entity label string

    Arguments:
        entity_label {str} -- entity label string

    Returns:
        tuple -- (chunk_tag, chunk_type)
    """
    res = re.match(r"^([^-]*)-(.*)$", entity_label)
    return res.groups() if res else (entity_label, "")


def start_of_chunk(pre_chunk_tag, pre_chunk_type, cur_chunk_tag, cur_chunk_type):
    """This function judges whether the start of chunk

    Arguments:
        pre_chunk_tag {str} -- previous chunk tag
        pre_chunk_type {str} -- previous chunk type
        cur_chunk_tag {str} -- current chunk tag
        cur_chunk_type {str} -- current chunk type

    Returns:
        bool -- the chunk is starting or not
    """
    # `O` must not be starting chunk
    if cur_chunk_tag == "O":
        return False

    # type of two consecutive chunks are different, current chunk must be starting chunk
    if cur_chunk_type != pre_chunk_type:
        return True

    # `B` and `U` must be starting chunk
    if cur_chunk_tag == "B" or cur_chunk_tag == "U":
        return True

    # before `I` and `E` must be `B` or `I`
    if cur_chunk_tag == "I" or cur_chunk_tag == "E":
        if pre_chunk_tag == "E" or pre_chunk_tag == "O" or pre_chunk_tag == "U":
            return True

    return False


def get_entity_span(entity_labels):
    """This function gets entity span

    Arguments:
        entity_labels {list} -- entity labels

    Returns:
        dict -- entity span index dict
    """
    pre_chunk_tag = "O"
    pre_chunk_type = ""
    chunk_list = []
    span2ent = defaultdict(str)

    for idx, entity_label in enumerate(entity_labels):
        cur_chunk_tag, cur_chunk_type = parse_entity_label(entity_label)
        is_start = start_of_chunk(
            pre_chunk_tag, pre_chunk_type, cur_chunk_tag, cur_chunk_type
        )

        if is_start:
            if chunk_list:
                span2ent[(chunk_list[1], chunk_list[-1] + 1)] = chunk_list[0]
            chunk_list = [cur_chunk_type, idx]
        elif chunk_list and cur_chunk_type == chunk_list[0]:
            chunk_list.append(idx)

        pre_chunk_tag = cur_chunk_tag
        pre_chunk_type = cur_chunk_type

    if chunk_list:
        span2ent[(chunk_list[1], chunk_list[-1] + 1)] = chunk_list[0]

    return span2ent


if __name__ == "__main__":
    span2ent = get_entity_span(
        [
            "B-1",
            "I-1",
            "U-1",
            "E-1",
            "I-1",
            "O",
            "B-1",
            "O",
            "O",
            "U-3",
            "E-2",
            "O",
            "I-1",
        ]
    )
    assert span2ent == {
        (0, 2): "1",
        (2, 3): "1",
        (3, 4): "1",
        (4, 5): "1",
        (6, 7): "1",
        (9, 10): "3",
        (10, 11): "2",
        (12, 13): "1",
    }
