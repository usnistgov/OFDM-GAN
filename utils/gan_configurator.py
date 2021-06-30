
class Automated_config_setup:
    """
    Class manages updating default GAN factor dictionary with updated values for factors modified in a screening study.
    """
    def __init__(self):
        self.temp = None

    def check_key_exists(self, dictionary, find_key):
        """
        Check if factor exists in nested
        :param dictionary: factor setting dictionary
        :param find_key: factor name
        :return:
        """
        if isinstance(dictionary, list):
            for d in dictionary:
                for result in self.check_key_exists(d, find_key):
                    yield result
        if isinstance(dictionary, dict):
            for k, v in dictionary.items():
                if find_key == k:
                    yield True
                if isinstance(v, dict):
                    for result in self.check_key_exists(v, find_key):
                        yield result
                elif isinstance(v, list):
                    for d in v:
                        for result in self.check_key_exists(d, find_key):
                            yield result

    def replace_value_original(self, dictionary, find_key, replace_val):
        """
        Replace defalut value with updated setting for nested dictionary
        :param dictionary: factor default setting dictionary
        :param find_key: factor name
        :param replace_val: updated value
        :return:
        """
        if find_key in dictionary:
            dictionary[find_key] = replace_val
            yield dictionary[find_key]
        for k in dictionary:
            if isinstance(dictionary[k], list):
                for i in dictionary[k]:
                    for j in self.replace_value(i, find_key, replace_val):
                        yield j

    def replace_value(self, dictionary, find_key, replace_val):
        """
        replace default value in sub-dictionary
        :param dictionary: factor default setting dictionary
        :param find_key: factor name
        :param replace_val: updated value
        :return:
        """
        if isinstance(dictionary, list):
            for d in dictionary:
                for result in self.replace_value(d, find_key, replace_val):
                    yield result
        if isinstance(dictionary, dict):
            for k, v in dictionary.items():
                if find_key == k:
                    dictionary[find_key] = replace_val
                    yield dictionary[find_key]
                if isinstance(v, dict):
                    for result in self.replace_value(v, find_key, replace_val):
                        yield result
                elif isinstance(v, list):
                    for d in v:
                        for result in self.replace_value(d, find_key, replace_val):
                            yield result

    def update_default_setting(self, dictionary, find_key, replace_val):
        """
        Update dictionary factor with new setting. searches nested dictionarys for any instance of factor key
        """
        if len(list(self.check_key_exists(dictionary, find_key))) != 0:
            list(self.replace_value(dictionary, find_key, replace_val))
            return dictionary
        else:
            return dictionary

    def map_params(self, single_config, dictionary):
        """
        Update default dictionary with a configuration of new settings for a subset of all dictionary factor keys.
        :param single_config:
        :param dictionary:
        :return:
        """
        for factor_name, factor_setting in zip(single_config.index, single_config):
            dictionary = self.update_default_setting(dictionary, factor_name, factor_setting)
        return dictionary
