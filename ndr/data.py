"""
Synthetic corpus of code-mixed WhatsApp NDR replies.

Why synthetic
-------------
There is no public labelled dataset of Hinglish NDR responses. In a real
deployment you would bootstrap with a few thousand hand-labelled replies pulled
from the WhatsApp BSP logs. This generator stands in for that, and is built to
be *adversarial to itself*: templates deliberately share vocabulary across
intents, and roughly a fifth of the rows are hard/ambiguous cases.

Honest caveat: metrics measured on this data are optimistic. Generated text has
bounded variety, and the noise model is not the same distribution as real human
typos. Treat the numbers as a smoke test of the architecture, not as an estimate
of production accuracy. See README for the plan to replace this.
"""

from __future__ import annotations

import random
import re

import pandas as pd

# --------------------------------------------------------------------------- #
# Slot vocabularies (shared across intents on purpose)
# --------------------------------------------------------------------------- #

GREET = ["", "", "", "hello ", "hi ", "bhai ", "sir ", "sir ji ", "madam ", "ji ",
         "namaste ", "hlo ", "arre ", "haan ", "ok "]
POLITE = ["", "", "please ", "plz ", "pls ", "kripya ", "krpya "]
TAIL = ["", "", "", "", " thanks", " thank u", " thnx", " ok", " pls", " plz",
        " jaldi", " urgent", " 🙏", " 🙏🙏", " ...", "!!", " dhanyavad", " shukriya"]

DAY = ["kal", "kl", "parso", "aaj", "sunday", "monday", "saturday", "sat",
       "somvar", "ravivar", "2 din baad", "3 din baad", "4 din bad",
       "agle hafte", "next week", "15 tarikh", "20 tarikh", "weekend pe",
       "tomorrow", "tmrw", "day after tomorrow"]
TOD = ["subah", "subh", "shaam", "sham ko", "dopahar", "raat ko", "morning",
       "evening", "11 baje", "5 baje", "6 bje", "10 am", "7 pm", "4 pm ke baad",
       "10 se 12", "2 se 5", "lunch ke baad", "office ke baad", "shaam 6 baje"]
PLACE = ["ghar", "ghr", "home", "office", "ofc", "flat", "society", "gaon",
         "hostel", "shop", "dukan", "hospital"]
CITY = ["delhi", "jaipur", "mumbai", "pune", "indore", "lucknow", "patna",
        "kota", "ajmer", "gurgaon", "noida", "bangalore"]
PARCEL = ["parcel", "order", "packet", "saman", "item", "product", "delivery",
          "shipment", "courier", "dabba"]
AGENT = ["delivery boy", "courier boy", "delivery wala", "dilivery boy",
         "delivery agent", "courier wala", "delivery person", "dlvry boy", "bhaiya"]
PHONE = ["9876543210", "8123456789", "7009876543", "+91 9812345678", "9001122334"]
PIN = ["302001", "110045", "400703", "560037", "226010"]

# --------------------------------------------------------------------------- #
# Templates per intent
# --------------------------------------------------------------------------- #

TEMPLATES: dict[str, list[str]] = {

    "CUSTOMER_UNAVAILABLE": [
        "{g}main {place} pe nahi tha{t}",
        "{g}hum log {place} pe nahi the us time{t}",
        "{g}{place} pe koi nahi tha{t}",
        "{g}koi ghar pe nahi hai abhi{t}",
        "{g}main office me tha uss waqt{t}",
        "{g}i was not at home{t}",
        "{g}nobody was there at home{t}",
        "{g}main bahar gaya tha thodi der ke liye{t}",
        "{g}sorry main bahar tha jab wo aaya{t}",
        "{g}mai duty pe tha{t}",
        "{g}hum market gaye the{t}",
        "{g}main {tod} tak office me rehta hu{t}",
        "{g}{tod} tak koi nahi milega ghar pe{t}",
        "{g}mai job pe jata hu subah to koi nahi hota{t}",
        "{g}sab log bahar the{t}",
        "{g}i was in a meeting could not pick{t}",
        "{g}mai so raha tha{t}",
        "{g}ghar pe tala tha{t}",
        "{g}hum shaadi me gaye the ek din ke liye{t}",
        "{g}main washroom me tha aur phone dur tha{t}",
        "{g}koi available nahi tha ghar par{t}",
        "{g}mai college me tha{t}",
        "{g}wo aaya tha par hum nahi the{t}",
        "{g}not available at that time{t}",
        "{g}mai {city} me nahi tha us din{t}",
        "{g}ghar pe koi nahi h {tod} ke pehle{t}",
        "{g}unavailable tha sorry{t}",
        "{g}mai bimar tha hospital gaya tha{t}",
    ],

    "RESCHEDULE_REQUEST": [
        "{g}{p}{day} bhej dena{t}",
        "{g}{p}{day} {tod} bhejo{t}",
        "{g}{p}deliver karo {day}{t}",
        "{g}{day} deliver kar dijiye{t}",
        "{g}{p}reschedule kar do {day} ke liye{t}",
        "{g}{p}please deliver tomorrow{t}",
        "{g}{p}{tod} bhejna aaj{t}",
        "{g}{day} tak rakh lo phir bhejna{t}",
        "{g}{p}next attempt {day} {tod} karo{t}",
        "{g}{day} {tod} ke baad hi bhejna{t}",
        "{g}can you deliver on {day}{t}",
        "{g}{p}{tod} ke baad aana{t}",
        "{g}mujhe {day} chahiye{t}",
        "{g}{day} bhej sakte ho kya{t}",
        "{g}{p}kal hi bhej do{t}",
        "{g}reschedule {day}{t}",
        "{g}{p}please try again {day}{t}",
        "{g}abhi mat bhejo {day} bhejna{t}",
        "{g}{tod} ka time rakh lo{t}",
        "{g}{day} {tod} me deliver kar dena main rahunga{t}",
        "{g}aaj nahi {day} kar do{t}",
        "{g}{p}{day} pe delivery schedule kar do{t}",
        "{g}next {day} works for me{t}",
        "{g}{p}{tod} me hi bhejna warna koi nahi milega{t}",
    ],

    "OUT_OF_STATION": [
        "{g}main {city} gaya hua hu 5 din ke liye{t}",
        "{g}hum out of station hai abhi{t}",
        "{g}mai gaon gaya hu wapas {day} aaunga{t}",
        "{g}i am travelling right now{t}",
        "{g}main tour pe hu {day} tak{t}",
        "{g}bahar hu shehar se bahar{t}",
        "{g}main {city} me hu abhi ghar pe nahi{t}",
        "{g}chhutti pe gaye hai family ke sath{t}",
        "{g}shaadi me gaye hai {city} wapas {day} aayenge{t}",
        "{g}out of city for a week{t}",
        "{g}mai {day} ko wapas aa raha hu tab bhejna{t}",
        "{g}hum log ghumne gaye hai{t}",
        "{g}abhi mai station se bahar hu 10 din{t}",
        "{g}vacation pe hu abhi{t}",
        "{g}main office ke kaam se dusre sheher me hu{t}",
        "{g}ghar band hai hum bahar gaye hai{t}",
        "{g}mai apne gaon me hu {day} tak nahi aaunga{t}",
        "{g}travelling till next week please hold{t}",
    ],

    "ADDRESS_INCORRECT": [
        "{g}address galat hai{t}",
        "{g}pata galat likha hai{t}",
        "{g}address incomplete hai flat number nahi hai{t}",
        "{g}mera address adhura hai{t}",
        "{g}wrong address mentioned{t}",
        "{g}pincode galat hai sahi {pin} hai{t}",
        "{g}address me galti hai sector galat h{t}",
        "{g}wo dusre area me chala gaya address galat tha{t}",
        "{g}house number missing hai isliye nahi mila{t}",
        "{g}address thik se nahi likha gaya{t}",
        "{g}ye mera address nahi hai{t}",
        "{g}landmark nahi diya isliye nahi dhundh paya{t}",
        "{g}address wrong hai correct hai {pin}{t}",
        "{g}galat pata h wo kahi aur pahuch gaya{t}",
        "{g}tower number galat hai{t}",
        "{g}address me pincode wrong hai please update karo{t}",
        "{g}delivery address incorrect please correct{t}",
        "{g}wo galat society me gaya tha{t}",
    ],

    "ADDRESS_CHANGE": [
        "{g}{p}address change kar do office pe bhejo{t}",
        "{g}naye address pe bhejna hai{t}",
        "{g}mai shift ho gaya hu naya address dunga{t}",
        "{g}please change address to my office{t}",
        "{g}ab {place} pe mat bhejo office pe bhejo{t}",
        "{g}address update karna hai{t}",
        "{g}ghar ki jagah office deliver kar do{t}",
        "{g}mera address badal gaya hai{t}",
        "{g}naya pincode {pin} hai wahan bhej do{t}",
        "{g}can you deliver to a different address{t}",
        "{g}address change karna hai kaise karu{t}",
        "{g}dusre pate pe bhej do {p}{t}",
        "{g}mai naye flat me shift hua hu address update karo{t}",
        "{g}office address pe deliver karo main {tod} tak wahi hu{t}",
        "{g}address change chahiye new address bhejta hu{t}",
    ],

    "COURIER_NO_CONTACT": [
        "{g}{agent} ne call hi nahi kiya{t}",
        "{g}koi call nahi aaya mujhe{t}",
        "{g}{agent} ne phone nahi kiya aur chala gaya{t}",
        "{g}no call received from {agent}{t}",
        "{g}usne call kiya par gate pe hi ruk gaya upar nahi aaya{t}",
        "{g}{agent} bola gate pe aao mai nahi aa sakta{t}",
        "{g}call kiya to uthaya nahi maine call back kiya to switch off{t}",
        "{g}{agent} ka number switch off aa raha hai{t}",
        "{g}mujhe koi phone call nahi aayi{t}",
        "{g}wo society ke bahar se hi wapas chala gaya{t}",
        "{g}delivery wale ne bina bataye wapas le gaya{t}",
        "{g}phone nahi kiya bas otp manga aur cut kar diya{t}",
        "{g}{agent} ne bola time nahi hai baad me aaunga phir nahi aaya{t}",
        "{g}koi contact nahi kiya kisi ne{t}",
        "{g}usne mujhe ring hi nahi ki{t}",
        "{g}delivery boy ka call nahi aaya warna mai le leta{t}",
        "{g}unreachable tha wo mujhse baat nahi hui{t}",
    ],

    "FAKE_ATTEMPT_DISPUTE": [
        "{g}koi aaya hi nahi tha{t}",
        "{g}{agent} aaya hi nahi phir bhi failed dikha rahe ho{t}",
        "{g}mai pura din ghar pe tha koi nahi aaya{t}",
        "{g}jhooth bol rahe hai koi nahi aaya tha{t}",
        "{g}nobody came to deliver{t}",
        "{g}fake attempt mark kiya hai{t}",
        "{g}no one visited my address{t}",
        "{g}galat update kiya hai koi delivery attempt nahi hua{t}",
        "{g}mai ghar pe hi tha koi nahi aaya ye farzi hai{t}",
        "{g}without coming they marked not delivered{t}",
        "{g}status galat update kiya hai koi nahi aaya{t}",
        "{g}ye jhoothi report hai mai {tod} se ghar pe hu{t}",
        "{g}guard bola koi nahi aaya tha{t}",
        "{g}fraud hai koi delivery attempt nahi kiya gaya{t}",
        "{g}aapke bande ne aaye bina hi cancel kar diya{t}",
        "{g}mai wait karta raha koi nahi aaya complaint karni hai{t}",
    ],

    "PAYMENT_ISSUE": [
        "{g}cash nahi tha ghar pe{t}",
        "{g}paise ki vyavastha nahi thi{t}",
        "{g}online payment kar sakta hu kya{t}",
        "{g}{p}payment link bhejo online kar dunga{t}",
        "{g}cod ki jagah upi se le sakta hu{t}",
        "{g}change nahi tha chhutte paise nahi the{t}",
        "{g}atm nahi gaya tha cash arrange nahi hua{t}",
        "{g}i want to pay online instead of cash{t}",
        "{g}cash on delivery nahi karna prepaid kar do{t}",
        "{g}paise kam pad gaye the{t}",
        "{g}gpay chalega kya cash nahi hai{t}",
        "{g}salary nahi aayi thi {day} tak paise aa jayenge{t}",
        "{g}amount jyada hai cash nahi tha{t}",
        "{g}phonepe kar dunga link bhej do{t}",
        "{g}cash ready nahi tha {day} ko cash rakh lunga{t}",
    ],

    "REFUSED_NOT_NEEDED": [
        "{g}mujhe ab ye {parcel} nahi chahiye{t}",
        "{g}bahut late ho gaya ab nahi chahiye{t}",
        "{g}maine galti se order kar diya tha{t}",
        "{g}i dont want this product now{t}",
        "{g}wapas bhej do mujhe nahi chahiye{t}",
        "{g}maine dusri jagah se le liya hai{t}",
        "{g}itna time lag gaya ab zarurat nahi{t}",
        "{g}mujhe nahi chahiye return kar dijiye{t}",
        "{g}refuse kar raha hu wapas le jao{t}",
        "{g}mera mann badal gaya nahi lena hai{t}",
        "{g}duplicate order ho gaya tha ye nahi chahiye{t}",
        "{g}price kam ho gaya hai dusri site pe nahi lunga{t}",
        "{g}not needed anymore return it{t}",
        "{g}quality achi nahi lagi nahi lena{t}",
        "{g}wapas le jao mujhe interest nahi hai{t}",
        "{g}gift ke liye tha ab kaam ka nahi raha{t}",
    ],

    "CANCEL_REQUEST": [
        "{g}{p}order cancel kar do{t}",
        "{g}cancel karna hai ye order{t}",
        "{g}please cancel my order{t}",
        "{g}{parcel} cancel kar dijiye refund chahiye{t}",
        "{g}cancel kro isko{t}",
        "{g}mujhe order cancel karna hai paise wapas kar do{t}",
        "{g}cancle kar do bhai{t}",
        "{g}kindly cancel this shipment{t}",
        "{g}order radd kar dijiye{t}",
        "{g}cancel and refund my amount{t}",
        "{g}ye order cancel kar ke paisa refund karo{t}",
        "{g}cancel plz mujhe nahi lena{t}",
    ],

    "ALREADY_RECEIVED": [
        "{g}mujhe {parcel} mil chuka hai{t}",
        "{g}ye to already deliver ho gaya tha{t}",
        "{g}maine kal hi receive kar liya tha{t}",
        "{g}i already received this order{t}",
        "{g}mera padosi ne le liya tha mujhe mil gaya{t}",
        "{g}guard ke paas rakha tha mil gaya{t}",
        "{g}already delivered phir kyu message aa raha hai{t}",
        "{g}mil gaya hai {parcel} status update karo{t}",
        "{g}delivered ho chuka hai mummy ne le liya tha{t}",
        "{g}ye galat message hai maine le liya tha{t}",
        "{g}parcel receive ho gaya hai kal shaam{t}",
    ],

    "SELF_PICKUP": [
        "{g}mai khud aake le lunga{t}",
        "{g}apke office se collect kar lunga address do{t}",
        "{g}hub ka address bhejo mai pick up kar lunga{t}",
        "{g}can i collect it from your warehouse{t}",
        "{g}mai branch se le aaunga timing bata do{t}",
        "{g}self pickup kar sakta hu kya{t}",
        "{g}godown kaha hai mai aa jaunga {day}{t}",
        "{g}mujhe khud lena hai delivery mat karo{t}",
        "{g}courier office se le lunga location share karo{t}",
        "{g}mai aake le lunga {tod} tak khula rehta hai kya{t}",
    ],

    "UNCLEAR": [
        "{g}ok{t}",
        "{g}hmm{t}",
        "{g}kya{t}",
        "{g}?{t}",
        "{g}haan{t}",
        "{g}thik hai{t}",
        "{g}kaun bol raha hai{t}",
        "{g}ye kya hai{t}",
        "{g}samajh nahi aaya{t}",
        "{g}kaunsa order{t}",
        "{g}maine to koi order nahi kiya{t}",
        "{g}stop sending messages{t}",
        "{g}achha{t}",
        "{g}...{t}",
        "{g}hello{t}",
        "{g}number kaha se mila aapko{t}",
        "{g}k{t}",
        "{g}😊{t}",
    ],
}

# --------------------------------------------------------------------------- #
# Hard / ambiguous cases — the ones that actually separate a good model
# from a demo. Written by hand, mixed into both train and test.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# English templates and slot vocabulary
# --------------------------------------------------------------------------- #
# Indian-English WhatsApp register: SMS shorthand, dropped apostrophes, missing
# articles, "kindly"/"do the needful" politeness. Deliberately NOT clean written
# English — that is not what arrives on a support line.

GREET_EN = ["", "", "", "hi ", "hello ", "hey ", "sir ", "madam ", "dear team ",
            "hi team ", "ok ", "hello sir "]
TAIL_EN = ["", "", "", "", " thanks", " thank you", " thanks a lot", " pls",
           " please", " tia", " regards", " kindly do the needful", " asap",
           " 🙏", "!!", " ok?"]
DAY_EN = ["tomorrow", "tmrw", "today", "monday", "sunday", "saturday",
          "next monday", "day after tomorrow", "in 2 days", "in 3 days",
          "next week", "this weekend", "on 15th", "on the 20th"]
TOD_EN = ["morning", "evening", "afternoon", "after 6pm", "before 11am",
          "after 5", "between 10 and 12", "around 4pm", "in the evening",
          "after office hours", "at 11 am", "by 7 pm", "post lunch"]

TEMPLATES_EN: dict[str, list[str]] = {

    "CUSTOMER_UNAVAILABLE": [
        "{g}i was not at home at that time{t}",
        "{g}nobody was available at the house{t}",
        "{g}i was in office when he came{t}",
        "{g}i had stepped out for some work{t}",
        "{g}we were all out for a function{t}",
        "{g}sorry i missed it i was travelling to work{t}",
        "{g}i work {tod} so nobody is there at home{t}",
        "{g}i was asleep did not hear the bell{t}",
        "{g}house was locked at that time{t}",
        "{g}i was in a meeting could not attend{t}",
        "{g}i had gone to drop my kids at school{t}",
        "{g}was not available please try again{t}",
        "{g}i am at work till {tod}{t}",
        "{g}we had gone to the market for 20 mins{t}",
        "{g}i was unwell and had gone to the doctor{t}",
        "{g}my family was not at home yesterday{t}",
    ],

    "RESCHEDULE_REQUEST": [
        "{g}please deliver {day}{t}",
        "{g}can you deliver it {day} {tod}{t}",
        "{g}kindly reschedule for {day}{t}",
        "{g}please try again {day}{t}",
        "{g}deliver {tod} i will be at home{t}",
        "{g}pls attempt delivery {day} instead{t}",
        "{g}i need it {day} only{t}",
        "{g}reschedule to {day} {tod} please{t}",
        "{g}not today please send {day}{t}",
        "{g}can u come {tod}{t}",
        "{g}please deliver after {tod}{t}",
        "{g}i am free {day} send it then{t}",
        "{g}shift the delivery to {day}{t}",
        "{g}next attempt should be {tod}{t}",
        "{g}hold it and deliver {day}{t}",
        "{g}request you to deliver {day} {tod}{t}",
    ],

    "OUT_OF_STATION": [
        "{g}i am out of station right now{t}",
        "{g}i am travelling will be back {day}{t}",
        "{g}currently i am not in the city{t}",
        "{g}i have gone to my home town for a week{t}",
        "{g}on vacation with family till {day}{t}",
        "{g}i am on an office tour to bangalore{t}",
        "{g}will return only after 10 days{t}",
        "{g}not in town please hold the parcel{t}",
        "{g}i am out of the country till next month{t}",
        "{g}gone to attend a wedding will be back {day}{t}",
        "{g}house is locked we are out of station{t}",
        "{g}i will be back in the city {day} only{t}",
    ],

    "ADDRESS_INCORRECT": [
        "{g}the address is wrong{t}",
        "{g}address mentioned is incomplete{t}",
        "{g}flat number is missing in the address{t}",
        "{g}wrong pincode it should be {pin}{t}",
        "{g}this is not my address{t}",
        "{g}the block written is incorrect{t}",
        "{g}house number is wrong in your record{t}",
        "{g}he went to a different locality{t}",
        "{g}street name is not correct{t}",
        "{g}address has a typo please correct it{t}",
        "{g}no landmark given so he could not find{t}",
        "{g}tower number mentioned is wrong{t}",
        "{g}the delivery address is incorrect kindly update{t}",
        "{g}my correct pincode is {pin} not what is shown{t}",
    ],

    "ADDRESS_CHANGE": [
        "{g}please change the address to my office{t}",
        "{g}i want it delivered at a different address{t}",
        "{g}i have shifted i will share the new address{t}",
        "{g}can you deliver to my office instead{t}",
        "{g}kindly update my delivery address{t}",
        "{g}deliver it to my shop not my home{t}",
        "{g}please send it to my brothers place{t}",
        "{g}new address pincode is {pin}{t}",
        "{g}address change required{t}",
        "{g}i moved last month please update the address{t}",
        "{g}send it to my office i am there till {tod}{t}",
        "{g}can the delivery location be changed{t}",
    ],

    "COURIER_NO_CONTACT": [
        "{g}the {agent} did not call me{t}",
        "{g}i did not receive any call{t}",
        "{g}no call was made before delivery{t}",
        "{g}he called once and cut the phone{t}",
        "{g}his number was switched off{t}",
        "{g}he refused to come up to my flat{t}",
        "{g}he asked me to come to the main gate{t}",
        "{g}i tried calling back but no response{t}",
        "{g}he never informed me he was outside{t}",
        "{g}the {agent} left without contacting me{t}",
        "{g}no one called otherwise i would have taken it{t}",
        "{g}he only asked for otp and disconnected{t}",
        "{g}i could not reach the delivery agent at all{t}",
    ],

    "FAKE_ATTEMPT_DISPUTE": [
        "{g}nobody came to my address{t}",
        "{g}no delivery attempt was made{t}",
        "{g}i was at home all day and no one came{t}",
        "{g}this is a false update{t}",
        "{g}they marked it failed without coming{t}",
        "{g}no one visited my place{t}",
        "{g}this is completely wrong no one turned up{t}",
        "{g}the security guard confirmed nobody came{t}",
        "{g}fake attempt has been marked on my order{t}",
        "{g}i waited the whole day and nobody showed up{t}",
        "{g}your agent never reached my building{t}",
        "{g}please check this is a false ndr{t}",
        "{g}i want to raise a complaint no attempt was made{t}",
    ],

    "PAYMENT_ISSUE": [
        "{g}i did not have cash at home{t}",
        "{g}can i pay online instead of cash{t}",
        "{g}please send me the payment link{t}",
        "{g}i want to convert this to prepaid{t}",
        "{g}cash was not arranged please give me a day{t}",
        "{g}i did not have change{t}",
        "{g}can i pay by upi or card{t}",
        "{g}the atm was not working{t}",
        "{g}i will pay through gpay share the link{t}",
        "{g}amount was more than the cash i had{t}",
        "{g}i will keep the cash ready by {day}{t}",
        "{g}please share a link i will transfer now{t}",
        "{g}cod is not possible for me let me pay online{t}",
    ],

    "REFUSED_NOT_NEEDED": [
        "{g}i do not want this product now{t}",
        "{g}it is too late i no longer need it{t}",
        "{g}i ordered it by mistake{t}",
        "{g}please return it to the seller{t}",
        "{g}i have already bought it from a local store{t}",
        "{g}not required anymore take it back{t}",
        "{g}i changed my mind i will not take it{t}",
        "{g}found it cheaper elsewhere so not needed{t}",
        "{g}i am refusing this delivery{t}",
        "{g}it was for a gift the occasion is over{t}",
        "{g}i am not accepting this parcel{t}",
        "{g}send it back i dont need it{t}",
    ],

    "CANCEL_REQUEST": [
        "{g}please cancel my order{t}",
        "{g}i want to cancel this order{t}",
        "{g}kindly cancel and refund the amount{t}",
        "{g}cancel this shipment please{t}",
        "{g}please close this order{t}",
        "{g}cancel it i will order again later{t}",
        "{g}requesting you to cancel the order{t}",
        "{g}cancel and process my refund{t}",
        "{g}i would like to cancel{t}",
    ],

    "ALREADY_RECEIVED": [
        "{g}i have already received this order{t}",
        "{g}it was delivered yesterday itself{t}",
        "{g}my wife collected it already{t}",
        "{g}the guard has already taken it for me{t}",
        "{g}i have it with me since yesterday{t}",
        "{g}this parcel is already delivered please check{t}",
        "{g}why am i getting this message i got it{t}",
        "{g}my neighbour received it on my behalf{t}",
        "{g}already got it kindly update the status{t}",
        "{g}the item is with me the status is wrong{t}",
    ],

    "SELF_PICKUP": [
        "{g}can i collect it from your office{t}",
        "{g}i will pick it up myself{t}",
        "{g}please share the hub address{t}",
        "{g}where is your warehouse i will come{t}",
        "{g}i prefer self pickup{t}",
        "{g}share the store location i will come{t}",
        "{g}what are your branch timings i will collect{t}",
        "{g}i can come and take it {day}{t}",
        "{g}send me the pickup location{t}",
        "{g}dont deliver i will collect it directly{t}",
    ],

    "UNCLEAR": [
        "{g}ok{t}",
        "{g}what{t}",
        "{g}who is this{t}",
        "{g}i did not order anything{t}",
        "{g}what is this message about{t}",
        "{g}please stop messaging me{t}",
        "{g}i dont understand{t}",
        "{g}which order are you talking about{t}",
        "{g}how did you get my number{t}",
        "{g}fine{t}",
        "{g}hmm{t}",
        "{g}?{t}",
    ],
}

HARD_CASES: list[tuple[str, str]] = [
    # unavailable vs fake-attempt (both mention "koi nahi") — negation scope matters
    ("ghar pe koi nahi tha isliye nahi mili", "CUSTOMER_UNAVAILABLE"),
    ("mai ghar pe hi tha koi nahi aaya", "FAKE_ATTEMPT_DISPUTE"),
    ("koi nahi aaya tha delivery karne", "FAKE_ATTEMPT_DISPUTE"),
    ("koi nahi tha ghar pe us waqt", "CUSTOMER_UNAVAILABLE"),
    # no-contact vs fake attempt
    ("aaya to tha par call nahi kiya", "COURIER_NO_CONTACT"),
    ("na call aaya na koi aaya", "FAKE_ATTEMPT_DISPUTE"),
    # cancel vs refuse
    ("nahi chahiye cancel kar do", "CANCEL_REQUEST"),
    ("mujhe nahi chahiye wapas le jao", "REFUSED_NOT_NEEDED"),
    # reschedule vs out of station
    ("kal bhej dena mai aa jaunga", "RESCHEDULE_REQUEST"),
    ("mai bahar hu kal wapas aaunga kal bhejna", "OUT_OF_STATION"),
    # address incorrect vs change
    ("address galat hai sahi wala bhejta hu", "ADDRESS_INCORRECT"),
    ("address sahi hai par office pe bhej do", "ADDRESS_CHANGE"),
    # payment vs cancel
    ("cash nahi hai isliye nahi liya kal le lunga", "PAYMENT_ISSUE"),
    ("itne paise nahi hai cancel kar do", "CANCEL_REQUEST"),
    # already received vs unclear
    ("mil gaya", "ALREADY_RECEIVED"),
    ("nahi mila", "UNCLEAR"),
    # pickup vs address change
    ("mai khud office aake le lunga", "SELF_PICKUP"),
    ("mere office pe bhej do", "ADDRESS_CHANGE"),
    # multi-intent (label = the action-bearing one)
    ("call nahi aaya tha kal shaam bhej dena", "RESCHEDULE_REQUEST"),
    ("address change karo aur kal bhejna", "ADDRESS_CHANGE"),
    # very short but decisive
    ("cancel", "CANCEL_REQUEST"),
    ("kal", "RESCHEDULE_REQUEST"),
    ("out of station", "OUT_OF_STATION"),
    ("wrong address", "ADDRESS_INCORRECT"),
    ("no cash", "PAYMENT_ISSUE"),
    ("nobody came", "FAKE_ATTEMPT_DISPUTE"),
    ("was at office", "CUSTOMER_UNAVAILABLE"),
    ("already got it", "ALREADY_RECEIVED"),
    # Devanagari
    ("मुझे नहीं चाहिए", "REFUSED_NOT_NEEDED"),
    ("कल भेज दीजिए", "RESCHEDULE_REQUEST"),
    ("पता गलत है", "ADDRESS_INCORRECT"),
    ("घर पर कोई नहीं था", "CUSTOMER_UNAVAILABLE"),
    ("ऑर्डर कैंसिल कर दो", "CANCEL_REQUEST"),

    # --- English hard cases: the same ambiguities, in the other language ------
    # 70% of production traffic is expected to be English, so the confusable
    # pairs need English coverage too — otherwise the model only learns the
    # distinctions in Hinglish and falls back to guessing in English.
    ("nobody was at home when he came", "CUSTOMER_UNAVAILABLE"),
    ("nobody came to deliver it", "FAKE_ATTEMPT_DISPUTE"),
    ("i was at home but no one came", "FAKE_ATTEMPT_DISPUTE"),
    ("no one was there to receive it", "CUSTOMER_UNAVAILABLE"),
    ("he came but did not call me", "COURIER_NO_CONTACT"),
    ("neither a call nor a visit", "FAKE_ATTEMPT_DISPUTE"),
    ("i dont want it cancel the order", "CANCEL_REQUEST"),
    ("i dont want it take it back", "REFUSED_NOT_NEEDED"),
    ("deliver tomorrow i will be here", "RESCHEDULE_REQUEST"),
    ("i am out of town back tomorrow send then", "OUT_OF_STATION"),
    ("the address is wrong i will send correct one", "ADDRESS_INCORRECT"),
    ("address is fine but deliver to office", "ADDRESS_CHANGE"),
    ("no cash right now i will take it tomorrow", "PAYMENT_ISSUE"),
    ("amount is too high please cancel", "CANCEL_REQUEST"),
    ("i will come to your office and take it", "SELF_PICKUP"),
    ("deliver it to my office address", "ADDRESS_CHANGE"),
    ("i have received it", "ALREADY_RECEIVED"),
    ("i have not received it", "UNCLEAR"),
    ("no call received but deliver tomorrow evening", "RESCHEDULE_REQUEST"),
    ("change my address and deliver tomorrow", "ADDRESS_CHANGE"),
    ("cancel", "CANCEL_REQUEST"),
    ("tomorrow", "RESCHEDULE_REQUEST"),
    ("wrong address", "ADDRESS_INCORRECT"),
    ("no cash", "PAYMENT_ISSUE"),
    ("out of station", "OUT_OF_STATION"),
    ("nobody came", "FAKE_ATTEMPT_DISPUTE"),
    ("was at office", "CUSTOMER_UNAVAILABLE"),
    ("already got it", "ALREADY_RECEIVED"),
    ("not needed", "REFUSED_NOT_NEEDED"),
    ("i will collect it", "SELF_PICKUP"),
]

# --------------------------------------------------------------------------- #
# Noise injection
# --------------------------------------------------------------------------- #

_KEYBOARD = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh",
    "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm",
    "o": "ip", "p": "o", "q": "wa", "r": "et", "s": "ad", "t": "ry", "u": "yi",
    "v": "cb", "w": "qe", "x": "zc", "y": "tu", "z": "x",
}
_VOWELS = "aeiou"


def _drop_vowel(w: str, rng: random.Random) -> str:
    idxs = [i for i, c in enumerate(w[1:], 1) if c in _VOWELS]
    return w[:idxs[rng.randrange(len(idxs))]] + w[idxs[rng.randrange(len(idxs))] + 1:] if idxs else w


def _typo(w: str, rng: random.Random) -> str:
    if len(w) < 3:
        return w
    i = rng.randrange(1, len(w))
    c = w[i]
    if c in _KEYBOARD:
        return w[:i] + rng.choice(_KEYBOARD[c]) + w[i + 1:]
    return w


def _swap(w: str, rng: random.Random) -> str:
    if len(w) < 4:
        return w
    i = rng.randrange(1, len(w) - 1)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def _stretch(w: str, rng: random.Random) -> str:
    if len(w) < 2:
        return w
    i = rng.randrange(len(w))
    return w[:i] + w[i] * rng.randint(2, 4) + w[i:]


def add_noise(text: str, rng: random.Random, rate: float = 0.22) -> str:
    """Corrupt tokens the way real thumb-typed WhatsApp text is corrupted."""
    ops = [_drop_vowel, _typo, _swap, _stretch]
    out = []
    for w in text.split():
        if len(w) > 2 and not w.isdigit() and rng.random() < rate:
            w = rng.choice(ops)(w, rng)
        out.append(w)
    s = " ".join(out)
    if rng.random() < 0.30:
        s = s.upper() if rng.random() < 0.2 else s.capitalize()
    if rng.random() < 0.15:
        s = s.replace(" ", "  ", 1)
    if rng.random() < 0.10:
        s = re.sub(r"[.,]", "", s)
    return s


# --------------------------------------------------------------------------- #
# Corpus builder
# --------------------------------------------------------------------------- #

def _fill(template: str, rng: random.Random, english: bool = False) -> str:
    return template.format(
        g=rng.choice(GREET_EN if english else GREET),
        p="" if english else rng.choice(POLITE),
        t=rng.choice(TAIL_EN if english else TAIL),
        day=rng.choice(DAY_EN if english else DAY),
        tod=rng.choice(TOD_EN if english else TOD),
        place=rng.choice(PLACE),
        city=rng.choice(CITY),
        parcel=rng.choice(PARCEL),
        agent=rng.choice(AGENT),
        phone=rng.choice(PHONE),
        pin=rng.choice(PIN),
    )


# Expected share of production replies written in English rather than Hinglish.
# This is a channel property, not a model property — measure it on real inbound
# traffic and set it accordingly. Getting it wrong is costly in one direction:
# whichever language is under-represented here becomes effectively
# out-of-vocabulary, and the model quietly guesses on it.
DEFAULT_ENGLISH_RATIO = 0.70


def build_corpus(per_intent: int = 320, seed: int = 7,
                 hard_case_copies: int = 12,
                 english_ratio: float = DEFAULT_ENGLISH_RATIO) -> pd.DataFrame:
    if not 0.0 <= english_ratio <= 1.0:
        raise ValueError("english_ratio must be between 0 and 1")

    rng = random.Random(seed)
    rows: list[dict] = []

    for intent in TEMPLATES:
        hi_templates = TEMPLATES[intent]
        en_templates = TEMPLATES_EN.get(intent, [])
        seen: set[str] = set()
        made = 0
        attempts = 0
        while made < per_intent and attempts < per_intent * 40:
            attempts += 1
            use_en = bool(en_templates) and rng.random() < english_ratio
            pool = en_templates if use_en else hi_templates
            base = _fill(rng.choice(pool), rng, english=use_en)
            text = add_noise(base, rng) if rng.random() < 0.75 else base
            text = text.strip()
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            rows.append({
                "text": text,
                "intent": intent,
                "source": "template",
                "lang": "en" if use_en else "hi",
            })
            made += 1

    for text, intent in HARD_CASES:
        for k in range(hard_case_copies):
            t = add_noise(text, rng, rate=0.18) if k else text
            # Devanagari and Roman-Hindi hard cases are tagged 'hi'; the rest
            # are heuristically English if they contain no Hinglish marker.
            hinglish_markers = (" nahi", " koi", " kar", " bhej", " ghar", " hai",
                                " tha", " mil ", " mujhe", " kal", " aaya")
            lang = "hi" if (any(m in f" {text.lower()}" for m in hinglish_markers)
                            or any("\u0900" <= c <= "\u097F" for c in text)) else "en"
            rows.append({"text": t, "intent": intent, "source": "hard", "lang": lang})

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = build_corpus()
    print(df["intent"].value_counts().to_string())
    print()
    print(df["lang"].value_counts(normalize=True).round(3).to_string())
    print()
    print(df.sample(15, random_state=1).to_string())
