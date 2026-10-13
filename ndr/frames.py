"""
Semantic frame bank for synthetic NDR reply generation.

Design claim
------------
A reply decomposes as:

    MESSAGE = [OPENER] + REASON + [CONSTRAINT] + [REQUEST] + [CLOSER]

Only REASON carries the intent. OPENER, CONSTRAINT, REQUEST and CLOSER are
intent-neutral modifiers that can attach to any reason. This is why the bank is
organised the way it is: intent-specific content is written once per intent,
and the combinatorial variety comes from composing it with shared modifiers.

Consequence for evaluation: a "frame" is (intent, reason_key). Frames are the
unit that must be disjoint between train and test, because two messages built
from the same reason_key are paraphrases, not independent samples.

Languages
    en  English (Indian-English register: 'kindly', 'do the needful', SMS forms)
    hr  Hinglish, Roman script
    hd  Hindi, Devanagari script
"""

LANGS = ("en", "hr", "hd")

# --------------------------------------------------------------------------- #
# Shared modifiers — intent-neutral
# --------------------------------------------------------------------------- #

OPENERS = {
    "en": ["", "", "", "hi ", "hello ", "sir ", "hi team ", "hello sir ",
           "dear team ", "sorry ", "actually ", "ok "],
    "hr": ["", "", "", "bhai ", "sir ", "ji ", "arre ", "namaste ", "hlo ",
           "sir ji ", "haan ", "sorry "],
    "hd": ["", "", "", "नमस्ते ", "सर ", "जी ", "माफ़ कीजिए ", "हैलो "],
}

CLOSERS = {
    "en": ["", "", "", " thanks", " thank you", " pls", " please", " tia",
           " regards", " kindly do the needful", " asap", "!!", " 🙏"],
    "hr": ["", "", "", " thanks", " thnx", " plz", " pls", " jaldi",
           " shukriya", " dhanyavad", "!!", " 🙏"],
    "hd": ["", "", "", " धन्यवाद", " कृपया", " शुक्रिया", " जल्दी", "!!", " 🙏"],
}

# CONSTRAINT: when / under what condition. Slot {d}=day, {t}=time.
CONSTRAINTS = {
    "en": ["", "", "", " {d}", " {t}", " {d} {t}", " only after {t}",
           " before {t}", " anytime {d}", " not before {t}"],
    "hr": ["", "", "", " {d}", " {t}", " {d} {t}", " {t} ke baad",
           " {t} se pehle", " {d} ko", " {t} ke baad hi"],
    "hd": ["", "", "", " {d}", " {t}", " {d} {t}", " {t} के बाद",
           " {t} से पहले", " {d} को"],
}

# REQUEST: what the customer wants done. Intent-neutral by design.
REQUESTS = {
    "en": ["", "", "", " please try again", " kindly reschedule",
           " please deliver then", " call me before coming",
           " please arrange accordingly", " do the needful",
           " let me know", " confirm once"],
    "hr": ["", "", "", " phir se try karna", " reschedule kar do",
           " tab bhej dena", " aane se pehle call karna",
           " arrange kar dijiye", " bata dena", " confirm kar do"],
    "hd": ["", "", "", " दोबारा कोशिश कीजिए", " रीशेड्यूल कर दीजिए",
           " तब भेज दीजिए", " आने से पहले कॉल कीजिए", " बता दीजिए"],
}

# --------------------------------------------------------------------------- #
# Slot vocabularies
# --------------------------------------------------------------------------- #

DAYS = {
    "en": ["tomorrow", "today", "monday", "sunday", "saturday", "next monday",
           "day after tomorrow", "in 2 days", "next week", "this weekend",
           "on 15th", "tmrw", "on the 20th"],
    "hr": ["kal", "aaj", "parso", "somvar", "ravivar", "shanivar", "kl",
           "2 din baad", "agle hafte", "weekend pe", "15 tarikh", "3 din bad"],
    "hd": ["कल", "आज", "परसों", "सोमवार", "रविवार", "शनिवार",
           "दो दिन बाद", "अगले हफ्ते", "15 तारीख"],
}

TIMES = {
    "en": ["morning", "evening", "afternoon", "6pm", "11am", "after 5",
           "between 10 and 12", "around 4pm", "after office hours", "by 7 pm"],
    "hr": ["subah", "shaam", "dopahar", "6 baje", "11 bje", "5 baje ke baad",
           "10 se 12", "shaam 6 baje", "office ke baad", "lunch ke baad"],
    "hd": ["सुबह", "शाम", "दोपहर", "6 बजे", "11 बजे", "शाम 6 बजे",
           "10 से 12", "दोपहर बाद"],
}

PLACES = {
    "en": ["home", "office", "the flat", "my shop", "the site"],
    "hr": ["ghar", "office", "flat", "dukan", "site"],
    "hd": ["घर", "ऑफिस", "फ्लैट", "दुकान"],
}

AGENTS = {
    "en": ["the delivery boy", "the courier agent", "the delivery person",
           "your executive", "the rider"],
    "hr": ["delivery boy", "courier wala", "delivery wala", "bhaiya",
           "aapka banda"],
    "hd": ["डिलीवरी बॉय", "कूरियर वाला", "डिलीवरी वाला", "आपका आदमी"],
}

PEOPLE = {
    "en": ["the security guard", "my neighbour", "my wife", "my brother",
           "my mother", "the receptionist", "my son"],
    "hr": ["guard", "padosi", "meri wife", "mere bhai", "mummy",
           "reception wale", "mere bete"],
    "hd": ["गार्ड", "पड़ोसी", "मेरी पत्नी", "मेरे भाई", "मम्मी", "रिसेप्शन वाले"],
}

CITIES = {
    "en": ["Delhi", "Mumbai", "Jaipur", "Pune", "Bangalore", "Lucknow",
           "Indore", "Kolkata", "my home town"],
    "hr": ["delhi", "mumbai", "jaipur", "pune", "bangalore", "lucknow",
           "gaon", "apne sheher"],
    "hd": ["दिल्ली", "मुंबई", "जयपुर", "पुणे", "बैंगलोर", "गाँव"],
}

PARCELS = {
    "en": ["the parcel", "the order", "the package", "the item", "the shipment"],
    "hr": ["parcel", "order", "packet", "saman", "shipment"],
    "hd": ["पार्सल", "ऑर्डर", "पैकेट", "सामान"],
}

# --------------------------------------------------------------------------- #
# REASON frames — the intent-bearing content
#
# Structure: INTENT -> reason_key -> lang -> [surface variants]
# Slots available: {d} day, {t} time, {place}, {agent}, {person}, {city},
#                  {parcel}, {pin}, {phone}, {flat}
# --------------------------------------------------------------------------- #

REASONS: dict[str, dict[str, dict[str, list[str]]]] = {

"CUSTOMER_UNAVAILABLE": {
  "at_work": {
    "en": ["i was at {place} when he came", "i was away at work at that time",
           "nobody is at {place} during working hours"],
    "hr": ["mai us waqt {place} me tha", "mai duty pe tha jab wo aaya",
           "working hours me {place} pe koi nahi hota"],
    "hd": ["मैं उस समय {place} में था", "मैं ड्यूटी पर था जब वो आया"]},
  "stepped_out": {
    "en": ["i had stepped out for a few minutes", "we had gone to the market",
           "i was out briefly when he arrived"],
    "hr": ["mai thodi der ke liye bahar gaya tha", "hum market gaye the",
           "bas 10 minute ke liye bahar tha"],
    "hd": ["मैं थोड़ी देर के लिए बाहर गया था", "हम बाज़ार गए थे"]},
  "nobody_home": {
    "en": ["nobody was at {place} at that time", "there was no one to receive it",
           "{place} was locked"],
    "hr": ["{place} pe koi nahi tha us waqt", "lene wala koi nahi tha",
           "{place} pe tala tha"],
    "hd": ["{place} पर कोई नहीं था उस समय", "लेने वाला कोई नहीं था"]},
  "asleep_missed": {
    "en": ["i was asleep and did not hear the bell",
           "i did not hear the doorbell", "i was upstairs and missed him"],
    "hr": ["mai so raha tha bell nahi suni", "darwaza nahi khol paya",
           "upar tha isliye pata nahi chala"],
    "hd": ["मैं सो रहा था घंटी नहीं सुनी", "दरवाज़ा नहीं खोल पाया"]},
  "occasion": {
    "en": ["we had all gone for a function", "there was a family event that day",
           "we were attending a wedding nearby"],
    "hr": ["hum sab ek function me gaye the", "us din ghar pe program tha",
           "shaadi me gaye the"],
    "hd": ["हम सब एक फंक्शन में गए थे", "उस दिन घर पर प्रोग्राम था"]},
  "unwell": {
    "en": ["i was unwell and had gone to the doctor",
           "i had to rush to the hospital"],
    "hr": ["mai bimar tha doctor ke pas gaya tha", "hospital jana pada tha"],
    "hd": ["मैं बीमार था डॉक्टर के पास गया था", "अस्पताल जाना पड़ा"]},
},

"RESCHEDULE_REQUEST": {
  "specific_day": {
    "en": ["please deliver {d}", "kindly attempt it {d}", "send it {d} instead"],
    "hr": ["{d} bhej dena", "{d} deliver kar dijiye", "{d} bhejna please"],
    "hd": ["{d} भेज दीजिए", "{d} डिलीवर कर दीजिए"]},
  "specific_slot": {
    "en": ["please come {t}", "deliver it {t}", "try {t} i will be at {place}"],
    "hr": ["{t} aana", "{t} bhej dena", "{t} try karna mai {place} pe rahunga"],
    "hd": ["{t} आइए", "{t} भेज दीजिए"]},
  "not_now": {
    "en": ["not today please try later", "do not send it now",
           "hold it and deliver later"],
    "hr": ["aaj nahi baad me bhejna", "abhi mat bhejo", "rok lo baad me dena"],
    "hd": ["आज नहीं बाद में भेजिए", "अभी मत भेजिए"]},
  "retry_request": {
    "en": ["please attempt the delivery once more", "can you try again",
           "kindly reattempt delivery"],
    "hr": ["ek baar phir se try karo", "dubara koshish kariye",
           "phir se bhejne ki koshish karo"],
    "hd": ["एक बार फिर कोशिश कीजिए", "दोबारा भेजिए"]},
  "availability_window": {
    "en": ["i am free only {t}", "i am at {place} only {t}",
           "i can take it only {d}"],
    "hr": ["mai sirf {t} free hu", "mai {place} pe sirf {t} rehta hu",
           "{d} hi le paunga"],
    "hd": ["मैं सिर्फ {t} फ्री हूँ", "{d} ही ले पाऊँगा"]},
},

"OUT_OF_STATION": {
  "travelling": {
    "en": ["i am out of station right now", "i am travelling at the moment",
           "i am not in the city currently"],
    "hr": ["mai abhi out of station hu", "mai travel kar raha hu",
           "abhi sheher me nahi hu"],
    "hd": ["मैं अभी शहर से बाहर हूँ", "मैं यात्रा पर हूँ"]},
  "at_city": {
    "en": ["i have gone to {city}", "i am in {city} for some work",
           "i am staying in {city} these days"],
    "hr": ["mai {city} gaya hua hu", "mai {city} me hu kaam se",
           "abhi {city} me rehta hu"],
    "hd": ["मैं {city} गया हुआ हूँ", "मैं {city} में हूँ काम से"]},
  "return_date": {
    "en": ["i will be back only {d}", "i return {d} please hold till then",
           "i reach back {d}"],
    "hr": ["mai {d} wapas aaunga", "{d} tak nahi aaunga tab tak rok lo",
           "{d} ko lautunga"],
    "hd": ["मैं {d} वापस आऊँगा", "{d} तक नहीं आऊँगा"]},
  "long_absence": {
    "en": ["we are away for about ten days", "the house is locked we have travelled",
           "nobody is in the flat we are all out"],
    "hr": ["hum 10 din ke liye bahar hai", "ghar band hai hum bahar gaye hai",
           "flat me koi nahi sab bahar hai"],
    "hd": ["हम दस दिन के लिए बाहर हैं", "घर बंद है हम बाहर गए हैं"]},
  "vacation": {
    "en": ["i am on vacation with family", "we have gone on a holiday trip"],
    "hr": ["mai family ke sath chhutti pe hu", "hum ghumne gaye hai"],
    "hd": ["मैं परिवार के साथ छुट्टी पर हूँ", "हम घूमने गए हैं"]},
},

"ADDRESS_INCORRECT": {
  "wrong_address": {
    "en": ["the address is wrong", "this is not my address",
           "the address on the label is incorrect"],
    "hr": ["address galat hai", "ye mera address nahi hai",
           "label pe pata galat likha hai"],
    "hd": ["पता गलत है", "यह मेरा पता नहीं है"]},
  "incomplete": {
    "en": ["the address is incomplete", "flat number is missing",
           "house number is not mentioned"],
    "hr": ["address adhura hai", "flat number nahi likha hai",
           "makan number missing hai"],
    "hd": ["पता अधूरा है", "फ्लैट नंबर नहीं लिखा है"]},
  "wrong_pincode": {
    "en": ["the pincode is wrong it should be {pin}",
           "wrong pin code mentioned my pin is {pin}"],
    "hr": ["pincode galat hai sahi {pin} hai", "pin galat h mera pin {pin} h"],
    "hd": ["पिनकोड गलत है सही {pin} है"]},
  "wrong_block": {
    "en": ["my flat is {flat} not what is written",
           "the block written is incorrect", "the tower number is wrong"],
    "hr": ["mera flat {flat} hai jo likha hai wo nahi",
           "block galat likha hai", "tower number galat hai"],
    "hd": ["मेरा फ्लैट {flat} है", "ब्लॉक गलत लिखा है"]},
  "misrouted": {
    "en": ["he went to a different locality", "it reached the wrong society",
           "no landmark given so he could not find it"],
    "hr": ["wo dusre area me chala gaya", "galat society me pahuch gaya",
           "landmark nahi tha isliye nahi dhundh paya"],
    "hd": ["वो दूसरे इलाके में चला गया", "गलत सोसाइटी में पहुँच गया"]},
},

"ADDRESS_CHANGE": {
  "to_office": {
    "en": ["please deliver it to my office instead",
           "send it to my {place} not {place}"],
    "hr": ["office pe bhej dena ghar ki jagah", "{place} pe bhej do"],
    "hd": ["ऑफिस पर भेज दीजिए", "{place} पर भेज दीजिए"]},
  "shifted": {
    "en": ["i have shifted to a new place", "i moved last month",
           "i relocated recently"],
    "hr": ["mai shift ho gaya hu", "maine ghar badal liya hai",
           "naye flat me aa gaya hu"],
    "hd": ["मैं शिफ्ट हो गया हूँ", "मैंने घर बदल लिया है"]},
  "different_address": {
    "en": ["please change the delivery address",
           "can you deliver to a different address", "i want it at another address"],
    "hr": ["address change kar dijiye", "dusre pate pe bhej sakte ho kya",
           "naye address pe chahiye"],
    "hd": ["पता बदल दीजिए", "दूसरे पते पर भेज सकते हैं क्या"]},
  "new_pincode": {
    "en": ["my new pincode is {pin} please update",
           "deliver to pincode {pin} now"],
    "hr": ["naya pincode {pin} hai update kar do", "ab {pin} pe bhejna"],
    "hd": ["नया पिनकोड {pin} है", "अब {pin} पर भेजिए"]},
},

"CONTACT_UPDATE": {
  "wrong_number": {
    "en": ["the number on the order is wrong", "you have an old number of mine",
           "the contact number saved is incorrect"],
    "hr": ["order pe number galat hai", "purana number pada hai aapke pas",
           "jo number save hai wo galat hai"],
    "hd": ["ऑर्डर पर नंबर गलत है", "पुराना नंबर पड़ा है आपके पास"]},
  "new_number": {
    "en": ["please use {phone} instead", "my new number is {phone}",
           "call me on {phone}"],
    "hr": ["{phone} pe call karna", "mera naya number {phone} hai",
           "is number pe baat karo {phone}"],
    "hd": ["{phone} पर कॉल कीजिए", "मेरा नया नंबर {phone} है"]},
  "unreachable": {
    "en": ["that number is switched off use another",
           "my phone was not working please note {phone}"],
    "hr": ["wo number band hai dusra le lo",
           "mera phone kharab tha {phone} note kar lo"],
    "hd": ["वो नंबर बंद है दूसरा ले लीजिए"]},
  "alternate_contact": {
    "en": ["please save {phone} as the alternate number",
           "note down this second number {phone}"],
    "hr": ["{phone} alternate number save kar lo",
           "ye dusra number likh lo {phone}"],
    "hd": ["{phone} वैकल्पिक नंबर सेव कर लीजिए"]},
},

"COURIER_NO_CONTACT": {
  "no_call": {
    "en": ["{agent} did not call me", "i did not receive any call",
           "no call came before the attempt"],
    "hr": ["{agent} ne call nahi kiya", "mujhe koi call nahi aayi",
           "aane se pehle koi phone nahi aaya"],
    "hd": ["{agent} ने कॉल नहीं किया", "मुझे कोई कॉल नहीं आई"]},
  "unreachable_agent": {
    "en": ["his number was switched off", "i called back but no response",
           "i could not reach {agent} at all"],
    "hr": ["uska number switch off tha", "maine call back kiya koi jawab nahi",
           "{agent} se baat hi nahi hui"],
    "hd": ["उसका नंबर बंद था", "मैंने कॉल बैक किया कोई जवाब नहीं"]},
  "refused_to_come": {
    "en": ["he asked me to come down to the gate",
           "he refused to come up to the flat",
           "he said he cannot come inside the society"],
    "hr": ["usne bola gate pe aao",
           "wo upar flat tak nahi aaya",
           "bola society ke andar nahi aa sakta"],
    "hd": ["उसने कहा गेट पर आओ", "वो ऊपर फ्लैट तक नहीं आया"]},
  "left_silently": {
    "en": ["he left without informing anyone",
           "he went back without telling me", "no intimation that he had arrived"],
    "hr": ["wo bina bataye chala gaya",
           "mujhe bataye bina wapas le gaya", "aane ki koi soochna nahi di"],
    "hd": ["वो बिना बताए चला गया", "मुझे बताए बिना वापस ले गया"]},
  "otp_only": {
    "en": ["he only asked for otp and disconnected",
           "he demanded otp on call and cut the phone"],
    "hr": ["sirf otp manga aur phone kaat diya",
           "call pe otp manga aur cut kar diya"],
    "hd": ["सिर्फ ओटीपी माँगा और फोन काट दिया"]},
},

"FAKE_ATTEMPT_DISPUTE": {
  "nobody_came": {
    "en": ["nobody came to my address", "no one visited my place",
           "not a single person came"],
    "hr": ["koi nahi aaya mere address pe", "koi bhi nahi aaya",
           "ek bhi banda nahi aaya"],
    "hd": ["कोई नहीं आया मेरे पते पर", "कोई भी नहीं आया"]},
  "was_home_all_day": {
    "en": ["i was at {place} the whole day and no one came",
           "i waited all day and nobody showed up"],
    "hr": ["mai pura din {place} pe tha koi nahi aaya",
           "din bhar wait kiya koi nahi aaya"],
    "hd": ["मैं पूरा दिन {place} पर था कोई नहीं आया"]},
  "false_update": {
    "en": ["this status is false", "they marked it failed without coming",
           "the update is wrong no attempt was made"],
    "hr": ["ye status jhootha hai", "bina aaye failed mark kar diya",
           "galat update hai koi attempt nahi hua"],
    "hd": ["यह स्टेटस झूठा है", "बिना आए फेल्ड मार्क कर दिया"]},
  "third_party_witness": {
    "en": ["{person} confirmed nobody came",
           "the gate register has no entry for your agent"],
    "hr": ["{person} ne bola koi nahi aaya",
           "gate register me koi entry nahi hai"],
    "hd": ["{person} ने बताया कोई नहीं आया"]},
  "complaint": {
    "en": ["this is a fake attempt please investigate",
           "i want to raise a complaint about this false ndr"],
    "hr": ["ye farzi attempt hai jaanch karo",
           "is jhoothi report ki complaint karni hai"],
    "hd": ["यह फर्जी अटेम्प्ट है जाँच कीजिए"]},
},

"PAYMENT_ISSUE": {
  "no_cash": {
    "en": ["i did not have cash at that time", "there was no cash at {place}",
           "i could not arrange the cash"],
    "hr": ["us waqt cash nahi tha", "{place} pe paise nahi the",
           "cash arrange nahi kar paya"],
    "hd": ["उस समय कैश नहीं था", "{place} पर पैसे नहीं थे"]},
  "wants_online": {
    "en": ["can i pay online instead", "i want to pay by upi",
           "let me pay digitally rather than cash"],
    "hr": ["online payment kar sakta hu kya", "upi se pay karna hai",
           "cash ki jagah online kar dunga"],
    "hd": ["क्या मैं ऑनलाइन भुगतान कर सकता हूँ", "यूपीआई से पे करना है"]},
  "wants_link": {
    "en": ["please send me the payment link",
           "share a link i will transfer now"],
    "hr": ["payment link bhej dijiye", "link bhejo mai abhi transfer karta hu"],
    "hd": ["पेमेंट लिंक भेज दीजिए", "लिंक भेजिए मैं अभी ट्रांसफर करता हूँ"]},
  "atm_issue": {
    "en": ["the atm was not working", "i could not withdraw money"],
    "hr": ["atm kaam nahi kar raha tha", "paise nikal nahi paya"],
    "hd": ["एटीएम काम नहीं कर रहा था", "पैसे नहीं निकाल पाया"]},
  "amount_short": {
    "en": ["the amount was more than the cash i had",
           "i did not have change for the full amount"],
    "hr": ["itne paise nahi the jitna amount tha",
           "chhutte paise nahi the"],
    "hd": ["उतने पैसे नहीं थे जितना अमाउंट था"]},
  "will_arrange": {
    "en": ["i will keep the cash ready {d}",
           "give me till {d} i will arrange the money"],
    "hr": ["{d} tak cash ready rakh lunga",
           "{d} tak paise arrange kar lunga"],
    "hd": ["{d} तक कैश तैयार रखूँगा"]},
},

"SELF_PICKUP": {
  "will_collect": {
    "en": ["i will collect it myself", "i prefer to pick it up",
           "do not deliver i will come and take it"],
    "hr": ["mai khud aake le lunga", "self pickup karna hai",
           "delivery mat karo mai aa jaunga"],
    "hd": ["मैं खुद आकर ले लूँगा", "सेल्फ पिकअप करना है"]},
  "asks_location": {
    "en": ["please share the hub address", "where is your warehouse",
           "send me the pickup location"],
    "hr": ["hub ka address bhej do", "aapka godown kaha hai",
           "pickup location share karo"],
    "hd": ["हब का पता भेज दीजिए", "आपका गोदाम कहाँ है"]},
  "asks_timing": {
    "en": ["what are your office timings", "till what time is it open",
           "when can i come and collect"],
    "hr": ["aapke office ka time kya hai", "kitne baje tak khula rehta hai",
           "kab aake le sakta hu"],
    "hd": ["आपके ऑफिस का समय क्या है", "कितने बजे तक खुला रहता है"]},
  "will_come_on": {
    "en": ["i can come {d} to collect it", "i will reach your branch {t}"],
    "hr": ["{d} aake le lunga", "{t} tak aapke branch pahunch jaunga"],
    "hd": ["{d} आकर ले लूँगा", "{t} तक आपके ब्रांच पहुँच जाऊँगा"]},
},

"ALTERNATE_RECIPIENT": {
  "leave_with_guard": {
    "en": ["please leave it with {person}", "hand it over to {person}",
           "you can give it to {person}"],
    "hr": ["{person} ke pas de dena", "{person} ko de dijiye",
           "{person} ke pas chhod do"],
    "hd": ["{person} के पास दे दीजिए", "{person} को दे दीजिए"]},
  "authorised_person": {
    "en": ["{person} will receive it on my behalf",
           "{person} is authorised to collect it"],
    "hr": ["{person} mere behalf pe le lenge",
           "{person} ko lene ka permission hai"],
    "hd": ["{person} मेरी तरफ से ले लेंगे"]},
  "at_reception": {
    "en": ["please drop it at the reception",
           "leave it at the society gate with {person}"],
    "hr": ["reception pe de dena",
           "society gate pe {person} ke pas chhod dena"],
    "hd": ["रिसेप्शन पर दे दीजिए"]},
  "someone_home": {
    "en": ["{person} will be at {place} {t} give it to them",
           "{person} is at {place} you can hand over"],
    "hr": ["{person} {t} {place} pe rahenge unko de dena",
           "{person} {place} pe hai unko de sakte ho"],
    "hd": ["{person} {t} {place} पर रहेंगे उन्हें दे दीजिए"]},
},

"ALREADY_RECEIVED": {
  "have_it": {
    "en": ["i have already received {parcel}", "{parcel} is already with me",
           "i got it already"],
    "hr": ["mujhe {parcel} mil chuka hai", "{parcel} mere pas aa gaya hai",
           "mil gaya hai already"],
    "hd": ["मुझे {parcel} मिल चुका है", "{parcel} मेरे पास आ गया है"]},
  "delivered_earlier": {
    "en": ["it was delivered {d} itself", "this reached me {d}",
           "it came {d} already"],
    "hr": ["{d} hi deliver ho gaya tha", "{d} mil gaya tha",
           "{d} hi aa gaya tha"],
    "hd": ["{d} ही डिलीवर हो गया था", "{d} मिल गया था"]},
  "proxy_received": {
    "en": ["{person} collected it for me", "{person} received it on my behalf",
           "it was handed to {person}"],
    "hr": ["{person} ne le liya tha mere liye", "{person} ne receive kar liya",
           "{person} ko de diya gaya tha"],
    "hd": ["{person} ने ले लिया था मेरे लिए", "{person} ने रिसीव कर लिया"]},
  "status_wrong": {
    "en": ["why am i getting this message i already have it",
           "please update the status it is delivered"],
    "hr": ["mujhe mil gaya phir ye message kyu aa raha hai",
           "status update kar do delivered ho chuka hai"],
    "hd": ["मुझे मिल गया फिर यह संदेश क्यों आ रहा है"]},
},

"REFUSED_NOT_NEEDED": {
  "too_late": {
    "en": ["it is too late now i do not need it",
           "this took too long i no longer want it"],
    "hr": ["bahut late ho gaya ab nahi chahiye",
           "itna time lag gaya ab zarurat nahi"],
    "hd": ["बहुत देर हो गई अब नहीं चाहिए"]},
  "bought_elsewhere": {
    "en": ["i already bought it from a local shop",
           "i got it cheaper somewhere else so not needed"],
    "hr": ["maine dukan se le liya hai",
           "sasta mil gaya kahi aur se isliye nahi chahiye"],
    "hd": ["मैंने दुकान से ले लिया है", "कहीं और से सस्ता मिल गया"]},
  "ordered_by_mistake": {
    "en": ["i ordered this by mistake", "this was an accidental order"],
    "hr": ["maine galti se order kar diya tha", "ye galti se order ho gaya"],
    "hd": ["मैंने गलती से ऑर्डर कर दिया था"]},
  "refuse_delivery": {
    "en": ["i am not accepting {parcel}", "please take it back",
           "send it back to the seller"],
    "hr": ["mai {parcel} nahi le raha", "wapas le jao",
           "seller ko wapas bhej do"],
    "hd": ["मैं {parcel} नहीं ले रहा", "वापस ले जाइए"]},
  "changed_mind": {
    "en": ["i changed my mind i will not take it",
           "i am no longer interested in this"],
    "hr": ["mera mann badal gaya nahi lunga",
           "ab isme interest nahi hai"],
    "hd": ["मेरा मन बदल गया नहीं लूँगा"]},
},

"CANCEL_REQUEST": {
  "cancel_plain": {
    "en": ["please cancel my order", "i want to cancel this order",
           "kindly cancel the order"],
    "hr": ["mera order cancel kar do", "ye order cancel karna hai",
           "order cancel kar dijiye"],
    "hd": ["मेरा ऑर्डर कैंसिल कर दीजिए", "यह ऑर्डर कैंसिल करना है"]},
  "cancel_refund": {
    "en": ["cancel it and refund my amount",
           "please cancel and process the refund"],
    "hr": ["cancel karke paisa refund kar do",
           "cancel kar dijiye aur refund kar dijiye"],
    "hd": ["कैंसिल करके पैसा रिफंड कर दीजिए"]},
  "close_order": {
    "en": ["please close this order", "terminate this order",
           "i wish to withdraw this purchase"],
    "hr": ["is order ko band kar do", "ye order khatam kar dijiye",
           "mai ye purchase wapas lena chahta hu"],
    "hd": ["इस ऑर्डर को बंद कर दीजिए"]},
  "cancel_and_reorder": {
    "en": ["cancel it i will order again later",
           "cancel this one i will place a fresh order"],
    "hr": ["cancel kar do baad me phir order kar lunga",
           "isko cancel karo mai naya order karunga"],
    "hd": ["कैंसिल कर दीजिए बाद में फिर ऑर्डर करूँगा"]},
},

"NEEDS_HUMAN_REVIEW": {
  "unintelligible": {
    "en": ["asdf", "....", "??", "hmm", "k", "ok", "hello", "yes", "sure"],
    "hr": ["kya", "haan", "hmm", "thik hai", "achha", "ji", "arre", "kya hua"],
    "hd": ["क्या", "हाँ", "ठीक है", "अच्छा", "जी", "क्या हुआ"]},
  "off_topic": {
    "en": ["who is this", "how did you get my number",
           "stop sending me these messages", "i did not order anything"],
    "hr": ["aap kaun bol rahe ho", "mera number kaha se mila",
           "ye message bhejna band karo", "maine koi order nahi kiya"],
    "hd": ["आप कौन बोल रहे हैं", "मेरा नंबर कहाँ से मिला",
           "यह संदेश भेजना बंद कीजिए"]},
  "legal_threat": {
    "en": ["i will file a case in consumer court",
           "i am going to take legal action against you",
           "i will report this to the consumer forum"],
    "hr": ["mai consumer court me case karunga",
           "aap logo pe legal action lunga",
           "consumer forum me complaint karunga"],
    "hd": ["मैं उपभोक्ता अदालत में केस करूँगा",
           "मैं कानूनी कार्रवाई करूँगा"]},
  "abusive_escalated": {
    "en": ["this is the third time this is unacceptable",
           "your service is pathetic i am extremely angry",
           "i have complained four times and nothing happened"],
    "hr": ["teesri baar ho raha hai ye bardasht se bahar hai",
           "aapki service bekaar hai mai bahut gussa hu",
           "char baar complaint ki kuch nahi hua"],
    "hd": ["तीसरी बार हो रहा है यह बर्दाश्त से बाहर है",
           "आपकी सेवा बेकार है मैं बहुत गुस्सा हूँ"]},
  "contradictory": {
    "en": ["cancel it no wait deliver it {d} actually cancel",
           "i want it but do not send it",
           "deliver today no do not deliver at all"],
    "hr": ["cancel kar do nahi ruko {d} bhej do phir cancel hi karo",
           "chahiye to hai par mat bhejo",
           "aaj bhej do nahi mat bhejo"],
    "hd": ["कैंसिल कर दीजिए नहीं रुकिए {d} भेज दीजिए फिर कैंसिल ही कीजिए"]},
  "damaged_or_wrong_item": {
    "en": ["the box was damaged so i did not take it",
           "wrong item was sent to me",
           "the packet was open and looked tampered"],
    "hr": ["dabba tuta hua tha isliye nahi liya",
           "galat item bheja gaya hai",
           "packet khula hua tha tampered lag raha tha"],
    "hd": ["डिब्बा टूटा हुआ था इसलिए नहीं लिया",
           "गलत सामान भेजा गया है"]},
},
}


def frame_inventory() -> list[tuple[str, str]]:
    """All (intent, reason_key) pairs — the unit of train/test disjointness."""
    return [(intent, key) for intent, reasons in REASONS.items() for key in reasons]


def frame_count() -> int:
    return len(frame_inventory())


def surface_count() -> int:
    return sum(len(v) for reasons in REASONS.values()
               for langs in reasons.values() for v in langs.values())
