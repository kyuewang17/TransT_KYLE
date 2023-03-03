import random
import numpy as np
import itertools

# Referred to https://sashamaps.net/docs/resources/20-colors/
__COLORS_DICT__ = {
    "red": (220, 0, 0),
    "blue": (0, 0, 230),
    "magenta": (240, 50, 230),
    "cyan": (70, 240, 240),
    "green": (60, 180, 75),
    "brown": (170, 110, 40),
    "maroon": (100, 0, 0),
    "orange": (245, 130, 48),
    "yellow": (230, 200, 0),
    "navy": (0, 0, 128),
    "skyblue": (0, 150, 220),
    "teal": (0, 128, 128),
    "black": (10, 10, 10)
}

__LINESTYLES_DICT__ = {
    "dotted": (0, (1, 1)),
    "loosely dashed": (0, (5, 10)),
    "dashed": (0, (5, 5)),
    "dashdotted": (0, (3, 5, 1, 5))
}

__MARKERS_DICT__ = {
    "circle": "o",
    "square": "s",
    "star": "*",
    "alphabet_x": "x",
    "thin_diamond": "d",

}

__MARKER_LIST__ = ["o", "x", "d"]


class PlotStyleObj(object):
    def __init__(self, **kwargs):
        _tmp_ = ["color", "marker", "linestyle"]

        # Unpack KWARGS
        is_color_normalize = kwargs.get("is_color_normalize", False)
        assert isinstance(is_color_normalize, bool)
        is_random_shuffle = kwargs.get("is_random_shuffle", False)
        assert isinstance(is_random_shuffle, bool)
        plt_style_activation_list = \
            kwargs.get("plt_style_activation_list", _tmp_)
        assert isinstance(plt_style_activation_list, (str, list, tuple))
        if isinstance(plt_style_activation_list, str):
            assert plt_style_activation_list in _tmp_
            plt_style_activation_list = [plt_style_activation_list]
        else:
            assert len(plt_style_activation_list) > 0
            assert set(plt_style_activation_list).issubset(set(_tmp_))

        # Generate Plot Style List , w.r.t. plot style activation modes
        plt_lists = [None] * len(plt_style_activation_list)
        for plt_idx, plt_feature in enumerate(plt_style_activation_list):
            if plt_feature == "color":
                colors_list = list(__COLORS_DICT__.values())
                if is_color_normalize is True:
                    _tmp_color_list = (np.array(colors_list) / 255.0).tolist()
                    colors_list = [tuple(c) for c in _tmp_color_list]
                plt_lists[plt_idx] = colors_list
            elif plt_feature == "marker":
                plt_lists[plt_idx] = list(__MARKERS_DICT__.values())
            elif plt_feature == "linestyle":
                plt_lists[plt_idx] = list(__LINESTYLES_DICT__.values())
            else:
                raise NotImplementedError()

        # Change order to make plot style as natural as possible
        try:
            ls_index = plt_style_activation_list.index("linestyle")
        except ValueError:
            ls_index = None
        try:
            mk_index = plt_style_activation_list.index("marker")
        except ValueError:
            mk_index = None
        try:
            c_index = plt_style_activation_list.index("color")
        except ValueError:
            c_index = None

        # Set Order
        _order = []
        _tmp_order = [ls_index, mk_index, c_index]
        for _order_idx in _tmp_order:
            if _order_idx is not None:
                _order.append(_order_idx)

        # Set Plot Lists in order
        plt_lists = [plt_lists[j] for j in _order]
        plt_style_activation_list = [
            plt_style_activation_list[j] for j in _order
        ]

        _plotStyle = list(itertools.product(*plt_lists))

        # Generate Plot Style List
        self.plotStyle = []
        for plt_style in _plotStyle:
            _dict = {}
            for j, v in enumerate(list(plt_style)):
                k = plt_style_activation_list[j]
                _dict[k] = v
            self.plotStyle.append(_dict)

        # Random Shuffle if Mode is ON
        if is_random_shuffle:
            random.shuffle(self.plotStyle)

        # # Fixed Shuffle if Mode is OFF (for neat visualization)
        # else:
        #     new_plotStyle = []
        #     idx_list = list(range(len(self.plotStyle)))
        #     cnt = 0
        #     while all([item == -1 for item in idx_list]) is False:
        #         if cnt % 2 == 0:
        #             j = cnt // 2
        #         else:
        #             j = len(self.plotStyle) - 1 - (cnt // 2)
        #         new_plotStyle.append(self.plotStyle[j])
        #         idx_list[j] = -1
        #         cnt += 1
        #     self.plotStyle = new_plotStyle

        # Iteration Counter
        self.__iter_counter = 0

    def __len__(self):
        return len(self.plotStyle)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            ret_val = self[self.__iter_counter]
        except IndexError:
            self.__iter_counter = 0
            raise StopIteration
        self.__iter_counter += 1
        return ret_val

    def __getitem__(self, idx):
        mod_idx = idx % len(self)
        return self.plotStyle[mod_idx]

    def __call__(self, idx):
        permute_idx = idx % len(self)
        return self.plotStyle[permute_idx]


if __name__ == "__main__":
    test_obj = PlotStyleObj(is_color_normalize=True, is_random_shuffle=False)

    print(3445)
