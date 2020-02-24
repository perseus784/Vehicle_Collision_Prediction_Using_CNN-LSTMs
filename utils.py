import json,os,re
import numpy as np
from config import *
from collections import Counter
class utils:

    def __init__(self,data_dir,indexing_file,mode):
        with open(indexing_file,"r") as vsi_file:
            self.vsi_obj=json.loads(vsi_file.read())
        self.vocabulary_dict=self.vsi_obj["vocabulary"]
        self.slot_dict=self.vsi_obj["slot"]
        self.intent_dict=self.vsi_obj["intent"]
        self.data = self.prepare_data(data_dir,max_len)
        self.indexed_data=self.index_data(self.data,indexing_file)


    def vocabulary_slot_intent_indexing(self,sentence,slots,intent):
        
        for i in ['<pad>','<unk>','<sos>','<eos>']:
            if i not in self.vocabulary_dict.keys():
                self.vsi_obj["vocabulary"][i]=len(self.vocabulary_dict)  
        for i in ['<pad>','<unk>','O']:
            if i not in self.slot_dict.keys():
                self.vsi_obj["slot"][i]=len(self.slot_dict)              
        
        for word in sentence:
            if word not in self.vocabulary_dict.keys():
                self.vsi_obj["vocabulary"][word]=len(self.vocabulary_dict)

        for slot in slots:
            if slot not in self.slot_dict.keys():
                self.vsi_obj["slot"][slot]=len(self.slot_dict)
        if intent not in self.intent_dict.keys():
            self.vsi_obj["intent"][intent]=len(self.intent_dict)

    def index_to_vocabulary(self,slot_list):
        vocab_list=[]
        for ind in slot_list:
            if ind==0:
                break
            for key,val in self.vocabulary_dict.items():
                if val==ind:
                    vocab_list.append(key)
        return vocab_list

    def index_to_slots(self,slot_list):
        vocab_list=[]
        for ind in slot_list:
            if ind==0:
                break
            for key,val in self.slot_dict.items():
                if val==ind:
                    vocab_list.append(key)
        return vocab_list

    def index_to_intent(self,ind):
        for key,val in self.intent_dict.items():
            if val==ind:
                return key
    
    def join_multiple_redundant_slots(self,sentence_list,slot_list):
        def duplicates(n): 
            counter=Counter(n) 
            dups=[i for i in counter if counter[i]!=1]
            result={}
            for item in dups:
                    result[item]=[i for i,j in enumerate(n) if j==item] 
            return result
        v_=len(duplicates(slot_list))
        for c in range(v_):
            val_dict=duplicates(slot_list)
            for k,v in val_dict.items():
                if len(k)>=4:
                    sentence_list[v[0]:v[-1]+1]=[' '.join(sentence_list[v[0]:v[-1]+1])]
                    slot_list[v[0]:v[-1]+1]=[slot_list[v[0]]]
                    break
        return sentence_list,slot_list

    def format_sentences(self,sentence):
        sentence=sentence.lower().rstrip()
        return re.sub("[-()#/@;':<>`+=~|.!?,]",'',sentence)

    def prepare_data(self,formatted_directory,max_sentence_length):
        formatted_data=[]
        dataset=[]
        for _file in os.listdir(formatted_directory):
            with open(os.path.join(formatted_directory,_file), "r") as json_file:
                temp_=json.loads(json_file.read())
                dataset+=temp_

        for data in dataset:
            sentence_list=["<pad>"]*max_sentence_length
            slot_list=sentence_list[:]
            sentence = data["sentence"][:]
            sentence.append("<eos>")
            slot=data["slot"][:]
            self.vocabulary_slot_intent_indexing(sentence,slot,data["intent"])
            sentence_list[:len(sentence)]=sentence
            slot_list[:len(slot)]=slot
            formatted_data.append([sentence_list,slot_list,data["intent"]]) 
        with open(vocabulary_slot_intent_index_json_file,"w") as vsi_out_file:
            json.dump(self.vsi_obj,vsi_out_file)  
        print("formmated the data")
     
        return formatted_data

    def index_data(self,formatted_data,indexing_file):
        indexed_data=[]
        with open(indexing_file,"r") as index_file:
            index_dict = json.loads(index_file.read())
        for data in formatted_data:
            indexed_sentence=[]
            indexed_slot=[]
            sentence,slots,intent=data
            for word in sentence:
                indexed_sentence.append(index_dict["vocabulary"][word])
            for slot in slots:
                indexed_slot.append(index_dict["slot"][slot])
            indexed_intent=index_dict["intent"][intent]
            true_length =sentence.index("<eos>")
            indexed_data.append([indexed_sentence,true_length,indexed_slot,indexed_intent])
        print("prepared indexed data")
        return indexed_data
    
    def convert_to_real(self,sentence_list,slot_list,intent_list):
        real_sentence=[]
        real_intent=[]
        real_slot=[]

        for i in sentence_list:
            d=[]
            for j in i:
                key_list=list(self.vocabulary_dict.keys())
                value_list=list(self.vocabulary_dict.values())
                d.append(key_list[value_list.index(j)])
            real_sentence.append(d)
        for i in intent_list:
            key_list=list(self.intent_dict.keys())
            value_list=list(self.intent_dict.values())
            real_intent.append(key_list[value_list.index(i)])
        
        for i in slot_list:
            d=[]
            for j in i:
                key_list=list(self.slot_dict.keys())
                value_list=list(self.slot_dict.values())
                d.append(key_list[value_list.index(j)])             
            real_slot.append(d)
        return real_sentence,real_intent,real_slot

    def batch_dispatch(self,batch_size):
        start_index=0
        end_index=batch_size
        while end_index < len(self.indexed_data):
            print("no of sentence complete.....",end_index)
            indexed_data=self.indexed_data[start_index:end_index]
            sentences_list=[]
            true_length_list=[]
            slot_list=[]
            intent_list=[]
            for i in indexed_data:
                sentences_list.append(i[0])
                true_length_list.append(i[1])
                slot_list.append(i[2])
                intent_list.append(i[3])
            start_index,end_index=end_index,end_index+batch_size
            sentences_list=np.array(list(map(list, zip(*sentences_list))))
            #slot_list=list(map(list, zip(*slot_list)))
            #print("dispatched data")
            yield [sentences_list,np.array(true_length_list),np.array(intent_list),np.array(slot_list)]
        
