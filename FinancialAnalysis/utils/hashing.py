import hashlib


def dict_hash(dict_: dict):
    """
    A utility method for hashing input parameters and producing a unique signature

    :param dict_: Parameters to hash, given as a dictionary

    :return: (str) The computed, SHA256 hash
    """

    sorted_items = tuple(sorted(dict_.items()))
    hash_fn = hashlib.sha256()

    for item in sorted_items:
        for sub_item in item:
            if type(sub_item) == str:
                encoded_param = sub_item.encode('utf-8')

            elif type(sub_item) == dict:
                keys = list(sub_item.keys())
                params = [sub_item[key] for key in keys]

                encoded_param = []
                for i, key in enumerate(keys):
                    if type(params[i]) == tuple or type(params[i]) == list:
                        encoded_param.append(key +
                                             ','.join([str(p) for p in params[i]]))

                    elif type(params[i]) == dict:
                        encoded_param.append(key +
                                             ','.join([str(k) + ',' + str(params[i][k])
                                                       for k in params[i]]))

                    else:
                        encoded_param.append(key + str(params[i]))

                encoded_param = ','.join(encoded_param).encode('utf-8')

            elif type(sub_item) == tuple or type(sub_item) == list:
                encoded_param = (','.join([str(p) for p in sub_item])).encode('utf-8')

            else:
                encoded_param = str(sub_item).encode('utf-8')

            hash_fn.update(encoded_param)

    hash_name = hash_fn.hexdigest()

    return hash_name
