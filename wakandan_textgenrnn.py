import requests
import zipfile
import io
import glob
from textgenrnn import textgenrnn
from datetime import datetime


def download_pytorch_example_data():
    """Download example data used in PyTorch's text generation RNN tutorial"""
    r = requests.get("https://download.pytorch.org/tutorial/data.zip")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()


def create_wakandan_name_dataset():
    """Dataset used for training and text generation"""
    wakandan_masculine_names = ["T'Challa", "T'Chaka", "W'Kabi", "N'Jobu", "N'Jadaka", "M'Baku", "Zuri", "Azzuri",
                                "T'Shan", "S'Yan", "N'Kano", "Q'Ran", "B'Tumba", "N'Baza", "T'Chanda", "N'Gamo", "N'Iix", "N'Tomo",
                                "W'Tambi", "D'Vanu", "T'Swana", "T'Kora", "N'Baku", "B'Gali", "G'Mal", "N'Gami", "D'arun",
                                "B'Ntro", "K'Shan", "K'Tyah", "M'Daka", "M'Wabu", "M'Gari", "M'Demwe", "N'Kono", "N'Baaka", "N'Yaga", "D'Kar", "H'Llah", "H'Rham", "K'Shamba",
                                "L'Ton", "M'Halak", "M'Jumbak", "M'Shula", "M'Toka", "N'Basa", "N'Dega", "N'Dele", "N'Gassi", "N'Lix", "T'Channa", "T'Charra", "T'Konda", "T'Dogo", "T'Wari"]

    wakandan_feminine_names = ["Okoye", "Nakia", "Shuri", "Ramonda", "Ayo", "Aneka", "Dalia", "Lulu", "Nailah",
                               "Nareema", "Onyeka", "Teela", "Asira", "Folami", "Mbali", "Zola", "Yami", "Zari", "Jahniss", "Amara",
                               "Nehanda", "Andebah", "Okusana", "Chandra", "Zawadi", "Onome", "Abena", "Asha", "Abeni", "Karota",
                               "Kisani", "Koni", "Mira", "Nanali", "Raki", "Shuriri", "Tanzika", "Thandiwe", "Xandra", "Xoliswa", "Zuni"]

    return (wakandan_masculine_names, wakandan_feminine_names)


def create_model(model_name="textgenrnn_model"):
    """Create a textgenrnn object that defines an underlying RNN model
    model_name - change to set file name of resulting trained models/texts
    """

    model = textgenrnn(name=model_name)

    return (model, model_name)


def train_model(model, dataset, new_model=True, num_epochs=60,  gen_epochs=5, train_size=0.7, dropout=0.2):
    """Train the RNN model on a given dataset."""
    model.reset()
    model.train_on_texts(dataset, new_model=True,
                         num_epochs=60,  gen_epochs=5, train_size=0.7, dropout=0.2)


def generate_name(model, return_as_list=True):
    """Generate a name from the trained model"""
    return model.generate(return_as_list=True)


def generate_names_to_file(model, model_name, num_of_names, prefix):
    """Generate names and print them to a file.
    num_of_names - number of names to generate and write to a file.
    prefix - a prefix for each of the names
    """
    # this temperature schedule cycles between 1 very unexpected token, 1 unexpected token, 2 expected tokens, repeat.
    # changing the temperature schedule can result in wildly different output!
    temperature = [1.0, 0.5, 0.2, 0.2]

    timestring = datetime.now().strftime('%Y%m%d_%H%M%S')
    gen_file = '{}_gentext_{}.txt'.format(model_name, timestring)

    model.generate_to_file(gen_file,
                           temperature=temperature,
                           prefix=prefix,
                           n=5,
                           max_gen_length=9)


def save_model():
    """# Saving the model

    When we trained the model, `textgenrnn` saved the weights, vocabulary, and configuration that resulted from the training to separate files.  Each of those files has a prefix of the `model_name` that we defined earlier in the code.

    We can download these files so that they can be loaded into a new `textgenrnn` model
    """
    pass
    # files.download('{}_weights.hdf5'.format(model_name))
    # files.download('{}_vocab.json'.format(model_name))
    # files.download('{}_config.json'.format(model_name))

def load_model(model_file, vocab_file, config_file):
    """Load a pre-trained model"""
    model = textgenrnn(weights_path=model_file, vocab_path=vocab_file, config_path=config_file)

    return model


def main():
    # download_pytorch_example_data()
    wakandan_masculine_names, wakandan_feminine_names = create_wakandan_name_dataset()
    # male_model, male_model_name = create_model("wakandan_masculine")
    # female_model, female_model_name = create_model("wakandan_feminine")

    # train_model(male_model, wakandan_masculine_names)
    # train_model(female_model, wakandan_feminine_names)
    model = load_model('wakandan_names_weights.hdf5', 'wakandan_names_vocab.json', 'wakandan_names_config.json')
    name = generate_name(model)



def get_model_api():
    '''Return lambda function for API'''
    # 1 Initilize model once and for all and reload weights

    # config_path = 'weights/RapLyrics_word2_01_config.json'
    # vocab_path = 'weights/RapLyrics_word2_01_vocab.json'
    # weights_path = 'weights/RapLyrics_word2_01_weights.hdf5'

    # textgen = textgenrnn(config_path=config_path,
    #                      vocab_path=vocab_path,
    #                      weights_path=weights_path)
    model = load_model('wakandan_names_weights.hdf5', 'wakandan_names_vocab.json', 'wakandan_names_config.json')
    model.generate() #resolved a memory addressing bug of keras, DO NOT remove

    def model_api():
        # # 2. pre-process input
        # punc = ["(", ")", "[",
        #         "]"]  # FIXME: add other cleaning if necessary, check if not redundant with library cleaning
        # prefix = "".join(c.lower() for c in input_data if c not in punc)

        # 3.0 initialize generation parameters
        # temperatures = [0.5, 0.6, 0.7] #TODO: to tweak.
        # num_line = 5
        # prefix_mode = 2  # see doc of sampler.py for mode 0,1,2
        # prefix_proba = 0.5

        # # 3.1 call model predict function
        # prediction = sampler.lyrics_generator(textgen, prefix,
        #                                       temperatures=temperatures, num_line=num_line,
        #                                       prefix_mode=prefix_mode, prefix_proba=prefix_proba)

        name = generate_name(model)
        # 4. process the output
        output_data = {"output": name}

        # 5. return the output for the api
        return output_data

    return model_api
    


if __name__ == "__main__":
    main()
