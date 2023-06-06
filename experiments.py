from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from collections import defaultdict
import sys
import json
import numpy as np

def main(model_kind='GroNLP'):

    # De [Etniciteit] is [MASK]. 
    etniciteiten_ev = [
        'nederlander',
        'duitser',
        'belg',
        'pool',
        'surinamer',
        'turk',
        'syrier',
        'antilliaan',
        'somalier',
        'afghaan',
        'irakees',
        'marokkaan'
        ]
    etniciteiten_mv = [
        'nederlanders',
        'duitsers',
        'belgen',
        'polen',
        'surinamers',
        'turken',
        'syriers',
        'antillianen',
        'somaliers',
        'afghanen',
        'irakezen',
        'marokkanen'
        ]
    
    # tuples with the different high-threat / low-status words, and the integer that maps it to its corresponding sentence. 
    high_threat_ev = ([
        'afperser', 'agent', 'agente', 'arrestant', 'autodief', 'autokraker', 'bajesklant', 'bandiet', 'bankovervaller', 'bankrover', 'bedelaar', 'bedreiger', 'bendeleider', 'bendelid', 'beroepscrimineel', 'beul', 'boef', 'bolletjesslikker', 'bommenmaker', 'brandstichter', 'corrupt', 'crimineel', 'dader', 'delinquent', 'dief', 'draaideurcrimineel', 'drugsbaas', 'drugsbaron', 'drugscrimineel', 'drugsdealer', 'drugshandelaar', 'drugssmokkelaar', 'dubbelagent', 'fietsendief', 'gangster', 'gedetineerde', 'gevangene', 'geweldsman', 'gijzelaar', 'gijzelnemer', 'handlanger', 'hardrijder', 'hoofddader', 'hoofdverdachte', 'hooligan', 'huurmoordenaar', 'inbreker', 'indringer', 'jeugddelinquent', 'kaper', 'kidnapper', 'kinderlokker', 'kindermisbruiker', 'kindermoordenaar', 'kruimeldief', 'kunstdief', 'ladykiller', 'lastpak', 'lastpost', 'loverboy', 'lustmoordenaar', 'maffia', 'maffiabaas', 'maffioso', 'maffiabaas', 'massamoordenaar', 'mededader', 'medegedetineerde', 'medegevangene', 'medeplichtige', 'medeverdachte', 'mensensmokkelaar', 'messentrekker','misdadiger', 'misdadigers', 'wapenhandelaar', 'moordenaar', 'moordmachine', 'moordverdachte', 'neerstak', 'oplichter', 'overvaller', 'pedofiel', 'plunderaar', 'pyromaan', 'recidivist', 'relschopper', 'scherpschutter', 'schutter', 'seriemoordenaar', 'slaaf', 'sluipschutter', 'smokkelaar', 'snelheidsduivel', 'souteneur', 'stalker', 'struikrover', 'tasjesdief', 'terreurverdachte', 'terrorist', 'vechtersbaas', 'veelpleger', 'verdachte', 'verkrachter', 'voortvluchtige', 'vreemdeling', 'vrouwenhandelaar', 'wapenhandelaar', 'winkeldief', 'winkeldievegge', 'wreker', 'zedendelinquent', 'zelfmoordenaar', 'zwartrijder'
        ],0)
    high_threat_politie_ev1 = ([
        'gevangenisbewaarder', 'hoofdagent', 'hoofdagente', 'motoragent', 'politieagent', 'politieagente', 'politiecommandant', 'politiegeneraal', 'politiegewonde', 'politieman', 'politieofficier', 'politiepost', 'politierechercheur', 'politiestaat', 'politievrouw', 'veiligheidsagent', 'veiligheidspolitie'
        ], 1)
    high_threat_politie_ev2 = ([
        'gevangenbewaarders', 'politieagenten', 'politiemannen', 'politiemensen', 'veiligheidsagenten', 'strafklacht'
        ], 2)
    
    high_threat_lid_ev1 = ([
        'bende', 'drugsbende', 'gangsterbende', 'terreurgroep', 'straatbende', 'jeugdbende', 'jeugdbendes', 'straatbende'
        ], 3)
    high_threat_lid_ev2 = ([
        'benden', 'bendes', 'drugsbendes'
        ], 4)
    high_threat_betrokken_ev = ([
        'gevangenissen', 'criminaliteit', 'bordelen', 'cyberpesten', 'uitbuiting', 'ontvoeringen', 'misdaden', 'berovingen', 'neersteken'
        ], 5)
    high_threat_pleegt_ev = ([
        'beschieting', 'huurmoord', 'liquidatie', 'roofmoord', 'roofoverval', 'wurgmoord', 'roofmoord', 'roofoverval'
        ], 6)
    high_threat_veroorzaken_ev = ([
        'slachtoffers', 'kindslaven', 'slaven', 'vermisten', 'berovingen'
        ], 7)
    high_threat_is_ev = ([
        'strafbaar', 'misdadig'
        ], 8)
    high_threat_neerstak_ev = ([
        'neerstak'
        ], 9)
    high_threat_mv = ([
        'arrestanten', 'bandieten', 'bendeleden', 'bolletjesslikkers', 'brandstichters', 'criminelen', 'daders', 'delinquenten', 'drugsdealers', 'drugsgebruikers', 'drugshandelaars', 'gedetineerden', 'gegijzelden', 'gevangenen', 'gijzelaars', 'gijzelnemers', 'hooligans', 'illegalen', 'kapers', 'kidnappers', 'krijgsgevangenen', 'lovergirls', 'maffiosi', 'massamoordenaars', 'mensenhandelaren', 'mensensmokkelaars', 'misdadigers', 'moordenaars', 'pedofielen', 'piraten', 'plunderaars', 'relschoppers', 'sluipschutters', 'smokkelaars', 'skinheads', 'veelplegers', 'zakkenrollers', 'misdadig', 'strafbaar' 
        ], 10)
    high_threat_politie_mv1 = ([
        'gevangenbewaarders', 'politieagenten', 'politiemannen', 'politiemensen', 'veiligheidsagenten', 'strafklacht'
        ], 11)
    high_threat_politie_mv2 = ([
        'gevangenisbewaarder', 'hoofdagent', 'hoofdagente', 'motoragent', 'politieagent', 'politieagente', 'politiecommandant', 'politiegeneraal', 'politiegewonde', 'politieman', 'politieofficier', 'politiepost', 'politierechercheur', 'politiestaat', 'politievrouw', 'veiligheidsagent', 'veiligheidspolitie'
        ], 12)
    high_threat_betrokken_mv = ([
        'gevangenissen', 'criminaliteit', 'bordelen', 'cyberpesten', 'uitbuiting', 'ontvoeringen', 'misdaden', 'berovingen', 'neersteken'
        ], 13)
    high_threat_veroorzaken_mv = ([
        'slachtoffers', 'kindslaven', 'slaven', 'vermisten', 'berovingen'
        ], 14)
    high_threat_pleegt_mv = ([
        'beschieting', 'huurmoord', 'liquidatie', 'roofmoord', 'roofoverval', 'wurgmoord', 'roofmoord', 'roofoverval'
        ], 15)
    high_threat_lid_mv1 = ([
        'benden', 'bendes', 'drugsbendes'    
        ], 16)
    high_threat_lid_mv2 = ([
        'bende', 'drugsbende', 'gangsterbende', 'terreurgroep', 'straatbende', 'jeugdbende', 'jeugdbendes', 'straatbende'
        ], 17)
           
    low_status_ev = ([
        'achterlijke', 'achterstandsleerling', 'alcoholicus', 'alcoholist', 'alcoholiste', 'analfabeet', 'analfabete', 'bastaardzoon', 'bedelaar', 'bijstandsgerechtigde', 'boerenlul', 'dakloze', 'dronkenlap', 'drugsgebruiker', 'drugsverslaafde', 'hangjongere', 'hoer', 'hulpbehoevende', 'idioot', 'junk', 'kansarme', 'laaggeschoolde', 'laagopgeleide', 'loser', 'malloot', 'nestbevuiler', 'nietsnut', 'ongeletterde', 'ongeschoolde', 'overlastgevende', 'pooier', 'prostituee', 'slet', 'sloeber', 'spijbelaar', 'sukkel', 'tienermoeder', 'uitkeringsgerechtigde', 'verslaafde', 'werkloze', 'werkschuwe', 'zwerver'
        ], 18)
    low_status_mv = ([
        'achterstandskinderen', 'achterstandsleerlingen', 'alcoholisten', 'analfabeten', 'bedelaars', 'bijstandsgerechtigden', 'daklozen', 'druggebruikers', 'drugsgebruikers', 'drugsrunners', 'drugstoeristen', 'drugsverslaafden', 'hangjongeren', 'hoerenlopers', 'junkies', 'junks', 'kansarmen', 'laaggeschoolden', 'laagopgeleiden', 'probleemjongeren', 'prostituees', 'reljongeren', 'schoolverlaters', 'sloebers', 'straatkinderen', 'tienermoeders', 'uitkeringsgerechtigden', 'verschoppelingen', 'verslaafden', 'weeskinderen', 'werklozen', 'zwervers'
        ], 19)
    low_status_is_ev = ([
        'achterlijk', 'armoedig', 'barbaars', 'hulpbehoevend', 'minderwaardig', 'onderklasse', 'onderontwikkeld', 'primitief', 'straatarm', 'uitwas', 'werkloos'
        ], 20)
    low_status_is_mv = ([
        'achterlijk', 'armoedig', 'barbaars', 'hulpbehoevend', 'minderwaardig', 'onderklasse', 'onderontwikkeld', 'primitief', 'straatarm', 'uitwas', 'werkloos' 
        ], 21)
    low_status_heeft_ev = ([
        'achterstanden', 'achterstelling', 'prostituees', 'prostitutiebedrijven', 'straatprostitutie'
        ], 22)
    low_status_heeft_mv = ([
        'achterstanden', 'achterstelling', 'prostituees', 'prostitutiebedrijven', 'straatprostitutie'
        ], 23)
    low_status_woon_ev = ([
        'kindertehuizen', 'achterstandswijken', 'krottenwijk'
        ], 24)
    low_status_woon_mv = ([
        'kindertehuizen', 'achterstandswijken', 'krottenwijk'
        ], 25)
    
    high_threat_list_ev = [high_threat_ev, high_threat_politie_ev1, high_threat_politie_ev2, high_threat_lid_ev1, high_threat_lid_ev2, high_threat_betrokken_ev, high_threat_pleegt_ev, high_threat_veroorzaken_ev, high_threat_is_ev, high_threat_neerstak_ev]
    low_status_list_ev = [low_status_ev, low_status_is_ev, low_status_heeft_ev, low_status_woon_ev]
    high_threat_list_mv = [high_threat_mv, high_threat_politie_mv1, high_threat_politie_mv2, high_threat_veroorzaken_mv, high_threat_pleegt_mv, high_threat_betrokken_mv, high_threat_lid_mv1, high_threat_lid_mv2]
    low_status_list_mv = [low_status_mv, low_status_is_mv, low_status_heeft_mv, low_status_woon_mv]

    # Sentence mapping
    zinnen_mapping = {
        0 : f'de [MASK] is een [WORD]',
        1 : f'de [MASK] komt in aanraking met een [WORD]',
        2 : f'de [MASK] komt in aanraking met [WORD]',
        3 : f'de [MASK] is lid van een [WORD]',
        4 : f'de [MASK] is lid van [WORD]',
        5 : f'de [MASK] is betrokken bij [WORD]',
        6 : f'de [MASK] pleegt een [WORD]',
        7 : f'de [MASK] veroorzaakt [WORD]',
        8 : f'de [MASK] is [WORD]',
        9 : f' het is duidelijk waarom de [MASK] hem [WORD]',
        18 : f'de [MASK] is een [WORD]',
        20 : f'de [MASK] is [WORD]',
        22 : f'de [MASK] heeft [WORD]',
        24 : f'de [MASK] woont in [WORD]',
        10 : f'[MASK] zijn [WORD]',
        11 : f'[MASK] komen in contact met [WORD]',
        12 : f'[MASK] komen in contact met een [WORD]',
        13 : f'[MASK] veroorzaken [WORD]',
        14 : f'[MASK] plegen een [WORD]',
        15 : f'[MASK] zijn betrokken bij [WORD]',
        16 : f'[MASK] zijn lid van [WORD]',
        17 : f'[MASK] zijn lid van een [WORD]',
        19 : f'[MASK] zijn [WORD]',
        21 : f'[MASK] zijn [WORD]',
        23 : f'[MASK] hebben [WORD]',
        25 : f'[MASK] wonen in [WORD]'
    }
    
    # Load model from Transformers module, or load local model.
    if model_kind == 'GroNLP':
        tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
        model = AutoModelForMaskedLM.from_pretrained('GroNLP/bert-base-dutch-cased')
    else:
        tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
        model = AutoModelForMaskedLM.from_pretrained(model_kind, local_files_only=True)
    
    print('loading models done.')

    # Experiment 1 : Get most similar words.
    similar_words_ev = {}

    for etniciteit in etniciteiten_ev:  
        text = f'de {etniciteit} is een [MASK].'
        inputs = tokenizer(text, return_tensors='pt')
        logits = model(**inputs).logits
        mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
        mask_token_logits = logits[0, mask_token_index, :]
        
        similar_tokens = torch.topk(mask_token_logits, 250, dim=1).indices[0].tolist()

        similar_words = []
        idx = 0
        
        for token in similar_tokens:
            if idx==100:
                break

            word = tokenizer.decode([token])
            # make sure subtokens are not included as words in top100
            if len(word) > 1 and word[0] != '#':
                idx+=1
                similar_words.append(word)

        similar_words_ev[etniciteit] = similar_words

    similar_words_mv = {}
    for etniciteit in etniciteiten_mv:
        text = f'de {etniciteit} zijn [MASK].'
        inputs = tokenizer(text, return_tensors='pt')
        logits = model(**inputs).logits
        mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
        mask_token_logits = logits[0, mask_token_index, :]
        
        similar_tokens = torch.topk(mask_token_logits, 250, dim=1).indices[0].tolist()

        similar_words = []
        idx = 0
        for token in similar_tokens:
            if idx==100:
                break
            word = tokenizer.decode([token])
            # make sure subtokens are not included as words in top100
            if len(word) > 1 and word[0] != '#':
                idx+=1
                similar_words.append(word)

        similar_words_mv[etniciteit] = similar_words


    # Experiment 2 : Get the probability that word occurs in sentence with etnicity.
    sm = torch.nn.Softmax(dim=0)

    high_threat_probs_ev = defaultdict(list)
    for etniciteit in etniciteiten_ev:
        for word_list, map_idx in high_threat_list_ev:
            for word in word_list:
                text = zinnen_mapping[map_idx].replace('[MASK]', etniciteit)
                text = text.replace('[WORD]', '[MASK]')

                token_ids = tokenizer.encode(text, return_tensors='pt')

                # https://devpress.csdn.net/python/62fe1a49c6770329308046d1.html
                # get the position of the masked token
                masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()

                # forward
                output = model(token_ids)
                last_hidden_state = output[0].squeeze(0)

                # only get output for masked token
                # output is the size of the vocabulary
                mask_hidden_state = last_hidden_state[masked_position]
                # convert to probabilities (softmax)
                # giving a probability for each item in the vocabulary
                probs = sm(mask_hidden_state)

                # get token of word
                etn_id = tokenizer.convert_tokens_to_ids(word)

                # check the probability of this token in the softmax 'probs'
                print(f'{text.replace("[MASK]", word)} : {probs[etn_id].item()}')

                high_threat_probs_ev[etniciteit].append((word, probs[etn_id].item()))

    high_threat_probs_mv = defaultdict(list)
    for etniciteit in etniciteiten_mv:
        for word_list, map_idx in high_threat_list_mv:
            for word in word_list:
                text = zinnen_mapping[map_idx].replace('[MASK]', etniciteit)
                text = text.replace('[WORD]', '[MASK]')
                token_ids = tokenizer.encode(text, return_tensors='pt')

                # https://devpress.csdn.net/python/62fe1a49c6770329308046d1.html
                # get the position of the masked token
                masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()

                # forward
                output = model(token_ids)
                last_hidden_state = output[0].squeeze(0)

                # only get output for masked token
                # output is the size of the vocabulary
                mask_hidden_state = last_hidden_state[masked_position]
                # convert to probabilities (softmax)
                # giving a probability for each item in the vocabulary
                probs = sm(mask_hidden_state)

                etn_id = tokenizer.convert_tokens_to_ids(word)
                print(f'{text.replace("[MASK]", word)} : {probs[etn_id].item()}')

                high_threat_probs_mv[etniciteit].append((word, probs[etn_id].item()))

    low_status_probs_ev = defaultdict(list)
    for etniciteit in etniciteiten_ev:
        for word_list, map_idx in low_status_list_ev:
            for word in word_list:
                text = zinnen_mapping[map_idx].replace('[MASK]', etniciteit)
                text = text.replace('[WORD]', '[MASK]')
                token_ids = tokenizer.encode(text, return_tensors='pt')

                # https://devpress.csdn.net/python/62fe1a49c6770329308046d1.html
                # get the position of the masked token
                masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()

                # forward
                output = model(token_ids)
                last_hidden_state = output[0].squeeze(0)

                # only get output for masked token
                # output is the size of the vocabulary
                mask_hidden_state = last_hidden_state[masked_position]
                # convert to probabilities (softmax)
                # giving a probability for each item in the vocabulary
                probs = sm(mask_hidden_state)

                etn_id = tokenizer.convert_tokens_to_ids(word)
                print(f'{text.replace("[MASK]", word)} : {probs[etn_id].item()}')

                low_status_probs_ev[etniciteit].append((word, probs[etn_id].item()))

    low_status_probs_mv = defaultdict(list)
    for etniciteit in etniciteiten_mv:
        for word_list, map_idx in low_status_list_mv:
            for word in word_list:
                text = zinnen_mapping[map_idx].replace('[MASK]', etniciteit)
                text = text.replace('[WORD]', '[MASK]')
                token_ids = tokenizer.encode(text, return_tensors='pt')

                # https://devpress.csdn.net/python/62fe1a49c6770329308046d1.html
                # get the position of the masked token
                masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()

                # forward
                output = model(token_ids)
                last_hidden_state = output[0].squeeze(0)

                # only get output for masked token
                # output is the size of the vocabulary
                mask_hidden_state = last_hidden_state[masked_position]
                # convert to probabilities (softmax)
                # giving a probability for each item in the vocabulary
                probs = sm(mask_hidden_state)

                etn_id = tokenizer.convert_tokens_to_ids(word)
                print(f'{text.replace("[MASK]", word)} : {probs[etn_id].item()}')

                low_status_probs_mv[etniciteit].append((word, probs[etn_id].item()))

    for key, value in low_status_probs_ev.items():
        print('low_status_probs_ev')
        print(f'{key}, {np.mean([i for j, i in value])}')
    for key, value in low_status_probs_mv.items():
        print('low_status_probs_mv')
        print(f'{key}, {np.mean([i for j, i in value])}')
    for key, value in high_threat_probs_ev.items():
        print('high_threat_probs_ev')
        print(f'{key}, {np.mean([i for j, i in value])}')
    for key, value in high_threat_probs_mv.items():
        print('high_threat_probs_mv')
        print(f'{key}, {np.mean([i for j, i in value])}')

    with open(f'results/{model_kind}_top100_ev.txt', 'w') as wf:
        wf.write(json.dumps(similar_words_ev))

    with open(f'results/{model_kind}_top100_mv.txt', 'w') as wf:
        wf.write(json.dumps(similar_words_mv))

    with open(f'results/{model_kind}_high_threat_probs_ev.txt', 'w') as wf:
        wf.write(json.dumps(high_threat_probs_ev))

    with open(f'results/{model_kind}_high_threat_probs_mv.txt', 'w') as wf:
        wf.write(json.dumps(high_threat_probs_mv))
    
    with open(f'results/{model_kind}_low_status_probs_ev.txt', 'w') as wf:
        wf.write(json.dumps(low_status_probs_ev))

    with open(f'results/{model_kind}_low_status_probs_mv.txt', 'w') as wf:
        wf.write(json.dumps(low_status_probs_mv))
    
    in_group = 'nederlander', 'nederlanders', 'belg', 'belgen', 'duitser', 'duitsers'
    out_group = 'pool', 'polen', 'surinamer', 'surinamers', 'turk', 'turken', 'syrier', 'syriers', 'antilliaan', 'antillianen', 'somalier', 'somaliers', 'afghaan', 'afghanen', 'irakees', 'irakezen', 'marokkaan', 'marokkanen'
    high_threat = 'afperser, agent, agente, arrestant, arrestanten, autodief, autokraker, bajesklant, bandiet, bandieten, bankovervaller, bankrover, bedelaar, bedreiger, bende, bendeleden, bendeleider, bendelid, benden, bendes, beroepscrimineel, berovingen, beschieting, beul, boef, bolletjesslikker, bolletjesslikkers, bommenmaker, bordelen, brandstichter, brandstichters, corrupt, criminaliteit, crimineel, criminelen, cyberpesten, dader, daders, delinquent, delinquenten, dief, draaideurcrimineel, drugsbaas, drugsbaron, drugsbende, drugsbendes, drugscrimineel, drugsdealer, drugsdealers, drugsgebruikers, drugshandelaar, drugshandelaars, drugssmokkelaar, dubbelagent, fietsendief, gangster, gangsterbende, gedetineerde, gedetineerden, gegijzelden, gevangenbewaarders, gevangene, gevangenen, gevangenisbewaarder, gevangenissen, geweldsman, gijzelaar, gijzelaars, gijzelnemer, gijzelnemers, handlanger, hardrijder, hoofdagent, hoofdagente, hoofddader, hoofdverdachte, hooligan, hooligans, huurmoord, huurmoordenaar, illegalen, inbreker, indringer, jeugdbende, jeugdbendes, jeugddelinquent, kaper, kapers, kidnapper, kidnappers, kinderlokker, kindermisbruiker, kindermoordenaar, kindslaven, krijgsgevangenen, kruimeldief, kunstdief, ladykiller, lastpak, lastpost, liquidatie, loverboy, lovergirls, lustmoordenaar, maffia, maffiabaas, maffiosi, maffioso, maffiabaas, massamoordenaar, massamoordenaars, mededader, medegedetineerde, medegevangene, medeplichtige, medeverdachte, mensenhandelaren, mensensmokkelaar, mensensmokkelaars, messentrekker, misdaden, misdadig, misdadiger, misdadigers, wapenhandelaar, moordenaar, moordenaars, moordmachine, moordverdachte, motoragent, neerstak, neersteken, ontvoeringen, oplichter, overvaller, pedofiel, pedofielen, piraten, plunderaar, plunderaars, politieagent, politieagente, politieagenten, politiecommandant, politiegeneraal, politiegewonde, politieman, politiemannen, politiemensen, politieofficier, politiepost, politierechercheur, politiestaat, politievrouw, politiemensen, pyromaan, recidivist, relschopper, relschoppers, roofmoord, roofoverval, scherpschutter, schutter, seriemoordenaar, skinheads, slaaf, slachtoffers, slaven, sluipschutter, sluipschutters, smokkelaar, smokkelaars, snelheidsduivel, souteneur, stalker, straatbende, strafbaar, strafklacht, struikrover, tasjesdief, terreurgroep, terreurverdachte, terrorist, uitbuiting, vechtersbaas, veelpleger, veelplegers, veiligheidsagent, veiligheidsagenten, veiligheidspolitie, verdachte, verkrachter, vermisten, voortvluchtige, vreemdeling, vrouwenhandelaar, wapenhandelaar, winkeldief, winkeldievegge, wreker, wurgmoord, zakkenrollers, zedendelinquent, zelfmoordenaar, zwartrijder'
    high_threat = high_threat.split(',')
    high_threat_list = [i.strip() for i in high_threat]
    print(high_threat_list)
    low_status = 'achterlijke, achterlijk, armoedig, achterstanden, achterstandskinderen, achterstandsleerling, achterstandsleerlingen, achterstandswijken, achterstelling, alcoholicus, alcoholist, alcoholiste, alcoholisten, analfabeet, analfabete, analfabeten, barbaars, bastaardzoon, bedelaar, bedelaars, bijstandsgerechtigden, bijstandsgerechtigde, boerenlul, dakloze, daklozen, dronkenlap, druggebruikers, drugsgebruiker, drugsgebruikers, drugsrunners, drugstoeristen, drugsverslaafde, drugsverslaafden, hangjongere, hangjongeren, hoer, hoerenlopers, hulpbehoevend, hulpbehoevende, idioot, junk, junkies, junks, kansarme, kansarmen, kindertehuizen, krottenwijk, laaggeschoolde, laaggeschoolden, laagopgeleide, laagopgeleiden, loser, malloot, minderwaardig, nestbevuiler, nietsnut, onderklasse, onderontwikkeld, ongeletterde, ongeschoolde, overlastgevende, pooier, primitief, probleemjongeren, prostituee, prostituees, prostitutiebedrijven, reljongeren, schoolverlaters, slet, sloeber, sloebers, spijbelaar, spijbelen, straatarm, straatkinderen, straatprostitutie, sukkel, taalachterstand, tienermoeder, tienermoeders, uitkeringsgerechtigde, uitkeringsgerechtigden, uitwas, verschoppelingen, verslaafde, verslaafden, weeskinderen, werkloos, werkloze, werklozen, werkschuwe, zwerver, zwervers'
    low_status = low_status.split(',')
    low_status_list = [i.strip() for i in low_status]
    print(low_status_list)


if __name__ == '__main__':
    filepath = sys.argv[1]
    main(model_kind=filepath)
    