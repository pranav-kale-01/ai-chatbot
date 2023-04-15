import pandas as pd 
from chatterbot.utils import print_progress_bar
from chatterbot.trainers import ListTrainer
from chatterbot import ChatBot
from os.path import exists

# reading dataset 
df = pd.read_csv( 'casual_data_windows.csv' )

conv_dataset = [] 

for index, row in df.iterrows(): 
    conv = []
    conv.append( row['0'])
    conv.append( row['1'])
    conv.append( row['2'])
    
    conv_dataset.append( conv )

    if index == 100: 
        break

db_exists = exists('db.sqlite3')

# creating instance of chatbot 
chatbot = ChatBot( 
    'CIH',
    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand.',
            'maximum_similarity_threshold': 0.90
        },
        {
            'import_path': 'chatterbot.logic.SpecificResponseAdapter',
            'input_text': 'What is your name',
            'output_text': 'I am CIH, i am the WAJIR of our team!'
        }, 
        {
            'import_path': 'chatterbot.logic.SpecificResponseAdapter',
            'input_text': 'Who is Atharv',
            'output_text': 'Atharv san inu desu'
        }
    ]
)

if not db_exists:
    trainer = ListTrainer( chatbot, show_training_progress=False  )

    # training using the reddit dataset
    dataset_len = len( conv_dataset )

    print("Dataset length : ", dataset_len )

    for k, conversation in enumerate( conv_dataset ): 
        print_progress_bar('training progress', k, dataset_len )
        trainer.train( conversation)    

    print()

while True: 
    # inputting the request statement
    request = input("You : ")

    if( str.lower(request) == 'exit' ): 
        break

    # Get a response to an input statement
    print( chatbot.name, ": ",chatbot.get_response(request), sep=' ' ) 

print("Thank you for using our chatbot! Hope you liked the experience")