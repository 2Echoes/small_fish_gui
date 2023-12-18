from .prompt import events


def post_analysis() :
    answer = events(['Save results', 'colocalisation', 'open results in napari'])

    return answer 