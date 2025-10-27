from abc import ABC, abstractmethod
from collections import namedtuple
import itertools
import math
from topicgpt_python.utils import *
import argparse
import re

# Metric Calculation copied from topicgpt_python which is based on: https://github.com/silviatti/topic-model-diversity/tree/master

em_topic_diversity = r"""
@article{DBLP:journals/corr/abs-1907-04907,
  author    = {Adji B. Dieng and
               Francisco J. R. Ruiz and
               David M. Blei},
  title     = {Topic Modeling in Embedding Spaces},
  journal   = {CoRR},
  volume    = {abs/1907.04907},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.04907},
  archivePrefix = {arXiv},
  eprint    = {1907.04907},
  timestamp = {Wed, 17 Jul 2019 10:27:36 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1907-04907.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

RBO = namedtuple("RBO", "min res ext")
RBO.__doc__ += ": Result of full RBO analysis"
RBO.min.__doc__ = "Lower bound estimate"
RBO.res.__doc__ = "Residual corresponding to min; min + res is an upper bound estimate"
RBO.ext.__doc__ = "Extrapolated point estimate"

def set_at_depth(lst, depth):
    ans = set()
    for v in lst[:depth]:
        if isinstance(v, set):
            ans.update(v)
        else:
            ans.add(v)
    return ans

def raw_overlap(list1, list2, depth):
    """Overlap as defined in the article.
    """
    set1, set2 = set_at_depth(list1, depth), set_at_depth(list2, depth)
    return len(set1.intersection(set2)), len(set1), len(set2)

def overlap(list1, list2, depth):
    """Overlap which accounts for possible ties.
    This isn't mentioned in the paper but should be used in the ``rbo*()``
    functions below, otherwise overlap at a given depth might be > depth which
    inflates the result.
    There are no guidelines in the paper as to what's a good way to calculate
    this, but a good guess is agreement scaled by the minimum between the
    requested depth and the lengths of the considered lists (overlap shouldn't
    be larger than the number of ranks in the shorter list, otherwise results
    are conspicuously wrong when the lists are of unequal lengths -- rbo_ext is
    not between rbo_min and rbo_min + rbo_res.
    >>> overlap("abcd", "abcd", 3)
    3.0
    >>> overlap("abcd", "abcd", 5)
    4.0
    >>> overlap(["a", {"b", "c"}, "d"], ["a", {"b", "c"}, "d"], 2)
    2.0
    >>> overlap(["a", {"b", "c"}, "d"], ["a", {"b", "c"}, "d"], 3)
    3.0
    """
    ov = agreement(list1, list2, depth) * min(depth, len(list1), len(list2))
    return ov
    # NOTE: comment the preceding and uncomment the following line if you want
    # to stick to the algorithm as defined by the paper
    # return raw_overlap(list1, list2, depth)[0]


def agreement(list1, list2, depth):
    """Proportion of shared values between two sorted lists at given depth.
    >>> _round(agreement("abcde", "abdcf", 1))
    1.0
    >>> _round(agreement("abcde", "abdcf", 3))
    0.667
    >>> _round(agreement("abcde", "abdcf", 4))
    1.0
    >>> _round(agreement("abcde", "abdcf", 5))
    0.8
    >>> _round(agreement([{1, 2}, 3], [1, {2, 3}], 1))
    0.667
    >>> _round(agreement([{1, 2}, 3], [1, {2, 3}], 2))
    1.0
    """
    len_intersection, len_set1, len_set2 = raw_overlap(list1, list2, depth)
    return 2 * len_intersection / (len_set1 + len_set2)

def rbo_min(list1, list2, p, depth=None):
    """Tight lower bound on RBO.
    See equation (11) in paper.
    >>> _round(rbo_min("abcdefg", "abcdefg", .9))
    0.767
    >>> _round(rbo_min("abcdefgh", "abcdefg", .9))
    0.767
    """
    depth = min(len(list1), len(list2)) if depth is None else depth
    x_k = overlap(list1, list2, depth)
    log_term = x_k * math.log(1 - p)
    sum_term = sum(
        p ** d / d * (overlap(list1, list2, d) - x_k) for d in range(1, depth + 1)
    )
    return (1 - p) / p * (sum_term - log_term)


def rbo_res(list1, list2, p):
    """Upper bound on residual overlap beyond evaluated depth.
    See equation (30) in paper.
    NOTE: The doctests weren't verified against manual computations but seem
    plausible. In particular, for identical lists, ``rbo_min()`` and
    ``rbo_res()`` should add up to 1, which is the case.
    >>> _round(rbo_res("abcdefg", "abcdefg", .9))
    0.233
    >>> _round(rbo_res("abcdefg", "abcdefghijklmnopqrstuvwxyz", .9))
    0.239
    """
    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l)
    # since overlap(...) can be fractional in the general case of ties and f
    # must be an integer --> math.ceil()
    f = int(math.ceil(l + s - x_l))
    # upper bound of range() is non-inclusive, therefore + 1 is needed
    term1 = s * sum(p ** d / d for d in range(s + 1, f + 1))
    term2 = l * sum(p ** d / d for d in range(l + 1, f + 1))
    term3 = x_l * (math.log(1 / (1 - p)) - sum(p ** d / d for d in range(1, f + 1)))
    return p ** s + p ** l - p ** f - (1 - p) / p * (term1 + term2 + term3)


def rbo_ext(list1, list2, p):
    """RBO point estimate based on extrapolating observed overlap.
    See equation (32) in paper.
    NOTE: The doctests weren't verified against manual computations but seem
    plausible.
    >>> _round(rbo_ext("abcdefg", "abcdefg", .9))
    1.0
    >>> _round(rbo_ext("abcdefg", "bacdefg", .9))
    0.9
    """
    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l)
    x_s = overlap(list1, list2, s)
    # the paper says overlap(..., d) / d, but it should be replaced by
    # agreement(..., d) defined as per equation (28) so that ties are handled
    # properly (otherwise values > 1 will be returned)
    # sum1 = sum(p**d * overlap(list1, list2, d)[0] / d for d in range(1, l + 1))
    sum1 = sum(p ** d * agreement(list1, list2, d) for d in range(1, l + 1))
    sum2 = sum(p ** d * x_s * (d - s) / s / d for d in range(s + 1, l + 1))
    term1 = (1 - p) / p * (sum1 + sum2)
    term2 = p ** l * ((x_l - x_s) / l + x_s / s)
    return term1 + term2


def rbo(list1, list2, p):
    """Complete RBO analysis (lower bound, residual, point estimate).
    ``list`` arguments should be already correctly sorted iterables and each
    item should either be an atomic value or a set of values tied for that
    rank. ``p`` is the probability of looking for overlap at rank k + 1 after
    having examined rank k.
    >>> lst1 = [{"c", "a"}, "b", "d"]
    >>> lst2 = ["a", {"c", "b"}, "d"]
    >>> _round(rbo(lst1, lst2, p=.9))
    RBO(min=0.489, res=0.477, ext=0.967)
    """
    if not 0 <= p <= 1:
        raise ValueError("The ``p`` parameter must be between 0 and 1.")
    args = (list1, list2, p)
    return RBO(rbo_min(*args), rbo_res(*args), rbo_ext(*args))


def get_word2index(list1, list2):
    words = set(list1)
    words = words.union(set(list2))
    word2index = {w: i for i, w in enumerate(words)}
    return word2index

class AbstractMetric(ABC):
    """
    Class structure of a generic metric implementation
    """

    def __init__(self):
        """
        init metric
        """
        pass

    @abstractmethod
    def score(self, model_output):
        """
        Retrieves the score of the metric

        :param model_output: output of a topic model in the form of a dictionary. See model for details on
        the model output
        :type model_output: dict
        """
        pass

    def get_params(self):
        return [att for att in dir(self) if not att.startswith("_") and att != 'info' and att != 'score' and
                att != 'get_params']

class TopicDiversity(AbstractMetric):
    def __init__(self, topk=10):
        """
        Initialize metric

        Parameters
        ----------
        topk: top k words on which the topic diversity will be computed
        """
        AbstractMetric.__init__(self)
        self.topk = topk

    def info(self):
        return {
            "citation": em_topic_diversity,
            "name": "Topic diversity"
        }

    def score(self, model_output):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        td : score
        """
        topics = model_output["topics"]
        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than ' + str(self.topk))
        else:
            unique_words = set()
            for topic in topics:
                unique_words = unique_words.union(set(topic[:self.topk]))
            td = len(unique_words) / (self.topk * len(topics))
            return td
        
class InvertedRBO(AbstractMetric):
    def __init__(self, topk=10, weight=0.9):
        """
        Initialize metric Inverted Ranked-Biased Overlap

        :param topk: top k words on which the topic diversity will be computed
        :param weight: weight of each agreement at depth d. When set to 1.0, there is no weight, the rbo returns to
        average overlap. (default 0.9)
        """
        super().__init__()
        self.topk = topk
        self.weight = weight

    def score(self, model_output):
        """
        Retrieves the score of the metric

        :param model_output : dictionary, output of the model. the 'topics' key is required.

        """
        topics = model_output['topics']
        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            collect = []
            for list1, list2 in itertools.combinations(topics, 2):
                word2index = get_word2index(list1, list2)
                indexed_list1 = [word2index[word] for word in list1]
                indexed_list2 = [word2index[word] for word in list2]
                rbo_val = rbo(indexed_list1[:self.topk], indexed_list2[:self.topk], p=self.weight)[2]
                collect.append(rbo_val)
            return 1 - np.mean(collect)

def metric_calc(data_file, ground_truth_col, output_col):
    """
    Calculate alignment metrics between predicted topics and ground-truth topics.

    Parameters:
    - data_file (str): Path to data file (containing both ground-truth and predicted topics)
    - ground_truth_col (str): Column name for ground-truth topics
    - output_col (str): Column name for predicted topics
    """
    # Load data
    data = pd.read_json(data_file, lines=True)
    output_topics = data[output_col]

    # Only retain the first topic in the list of topics
    output_pattern = r"\[(?:\d+)\] ([^:]+): (?:.+)"
    output_topics = [re.findall(output_pattern, topic)[0] for topic in output_topics]

    data["parsed_output"] = output_topics

    harmonic_purity, ari, mis = calculate_metrics(
        ground_truth_col, "parsed_output", data
    )

    print("--------------------")
    print("Alignment between predicted topics and ground truth:")
    print("Harmonic Purity: ", harmonic_purity)
    print("ARI: ", ari)
    print("MIS: ", mis)
    print("--------------------")

    return calculate_metrics(ground_truth_col, "parsed_output", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate alignment metrics between topics and ground-truth topics."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/input/assignment.jsonl",
        help="Path to data file (containing both ground-truth and predicted topics)",
    )
    parser.add_argument(
        "--ground_truth_col",
        type=str,
        default="ground_truth",
        help="Column name for ground-truth topics",
    )
    parser.add_argument(
        "--output_col",
        type=str,
        default="output",
        help="Column name for predicted topics",
    )
    args = parser.parse_args()

    metric_calc(args.data_file, args.ground_truth_col, args.output_col)
