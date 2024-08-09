import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI
import os
import requests

# Define paths
faiss_index_path = "faiss_index.bin"
chunks_path = "chunks.pkl"
databricks_token = "dapi231e912aa93cedec276065cb8995cce5"
server_endpoint = "https://adb-1769509101973077.17.azuredatabricks.net/serving-endpoints"

# GitHub raw URLs
github_base_url = "https://raw.githubusercontent.com/chiefmanish/Streamlit_pdf/main/"

# Download FAISS index
if not os.path.exists(faiss_index_path):
    response = requests.get(github_base_url + faiss_index_path)
    with open(faiss_index_path, "wb") as f:
        f.write(response.content)

# Download chunks
if not os.path.exists(chunks_path):
    response = requests.get(github_base_url + chunks_path)
    with open(chunks_path, "wb") as f:
        f.write(response.content)

# Load FAISS index
index = faiss.read_index(faiss_index_path)

# Load text chunks
with open(chunks_path, "rb") as f:
    all_chunks = pickle.load(f)

# Load the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to get the most relevant chunks
def get_relevant_chunks(query, index, model, chunks, top_k=10):
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    return relevant_chunks

# Function to get results from the Databricks model
def get_result(databricks_token, server_endpoint, extracted_text, question):
    client = OpenAI(
        api_key=databricks_token,
        base_url=server_endpoint
    )
    response = client.chat.completions.create(
        model="databricks-dbrx-instruct",
        messages=[

            {
                "role":"system",
                "content": f"""
                
                
                This assistant is a chat bot to answer spiritual questions. Respond to questions only if the answer can be found within the content provided. And then build your answer only from the content and while developing your answer from the content you can be creative and try to respond which should be a profound and contemplative explanation, reflecting the depth and insight typical of Osho's teachings. It should not be in a conversational format with a seeker. Instead, provide a reflective and instructive answer, using examples or metaphors as needed, all strictly based on the provided content.
                When answering, refer to yourself as Osho, embodying his wisdom and insight. Think as if you are the one who has written the content and must frame the answer based strictly on the knowledge provided in the context.
                If the question is not related to the provided content, respond with:
                "Ah, I see where you’re coming from, but this question is not covered in the wisdom at hand. Please ask a question related to the provided content."
                don't try to respond that is not asked by the user.
                user
                When answering, refer to yourself as Osho, embodying his wisdom and insight. Think as if you are the one who has written the content and must frame the answer based strictly on the knowledge provided below:
                It is one of the trade secrets of all the religions to propose propaganda that humanity has
                to be saved.
                It is a very strange idea, but it is so old that nobody seems to look into the implications.
                Nobody asks why you are worried about saving humanity. And you have been saving
                humanity for thousands of years, but nothing seems to be saved.
                In the first place, does humanity need any saving?
                To answer this question all the religions have created an absolutely fictitious idea of the
                original fall, because unless there is a fall the question of saving does not arise. And the
                religious conception of the original fall is just rubbish.
                Man has been evolving -- not falling -- in every possible way. The only way the original
                fall can be supported is by the idea of evolution proposed by Charles Darwin; but religions
                cannot use that -- they are very much offended. Charles Darwin's idea certainly can be put in
                such a way -- at least by the monkeys if not by man -- that it was an original fall. Certainly if
                man has evolved out of monkeys he must have fallen from the trees, and the monkeys who
                did not fall must have laughed at these idiots who had fallen. And there is a possibility that
                these were the weaker monkeys who could not survive in the trees.
                In monkeys there exists a hierarchy. Perhaps the same mind and the same hierarchy are
                carried by man too; it is the same mind. If you see monkeys sitting in a tree you can know
                who's the chief: he will be at the top of the tree. Then there will be a big group of ladies, his
                harem -- the most beautiful, young. After that will be a third group.
                I was thinking about this third group for many days but I had no word for it. In India we
                call that group the chamchas. Chamcha means a spoon, and these people are suckers. Just the
                way you take, with a spoon, things out of a bottle, they go on taking things -- power, money
                -- from those who have. Of course, they have to buttress these people, they have to praise
                these people.
                But Devaraj has sent by coincidence today the right word -- because chamcha cannot be
                exactly translated; "spoon" loses all meaning. He has sent me a word which is Californian:
                the brownnose. And he sent me the Webster's dictionary also because I might not understand
                what a brownnose is; and certainly I would not have understood what a brownnose is.
                He sent a note also, thinking perhaps that even the dictionary may not be helpful because
                Webster writes it in such a way that it does not look in any way obscene, dirty. So he sent me
                a note also: "In Europe we call these people `arse-kissers'." That's exactly the meaning of
                chamchas.
                The chief on the top, then the harem of the ladies whom he controls, then the
                brownnoses! And then you come down to lower categories of the hierarchy. On the lowest
                branches are the poorest monkeys, without girlfriends, boyfriends -- servants. But perhaps
                from this very group humanity has grown.
                Even in this group there may have been a few people who were so weak that they could
                not even manage to stay on the lowest branches. They were pushed, pulled, thrown, and
                somehow they found themselves fallen onto the earth. That is the original fall.
                Monkeys still go on laughing at man. Certainly if you think from the monkey's side, a
                monkey walking on two legs ... if you are a monkey and you think from its side, seeing a
                monkey walking on two legs, you will think, "Has he joined a circus or something? And what
                happened to the poor guy? He just lives on the ground; he never comes to the trees, the wild
                freedom of the trees, the higher status of the trees. This is really the fallen one, the
                downtrodden."
                Except for this, religions don't have any logical support for the idea of the original fall.
                Stories they have, but stories are not arguments, stories are not proofs. And stories can have
                just the opposite meaning to that which you wanted to give to them. For example, the original
                fall in Christianity makes God the real culprit, and if anybody needs saving it is the Christian
                God.
                A father preventing his children from being wise, from living forever, is certainly insane.
                Even the worst father would like his children to be wise, intelligent. Even the cruelest father
                would like his children to live forever.
                But God prevents man from eating of two trees -- the tree of knowledge and the tree of
                eternal life. This seems to be a strange kind of God; it is not in any way possible to conceive
                Him as fatherly. He seems to be the enemy of man. Who needs saving? Your God is jealous:
                that's what was the argument of the devil who came in the form of a serpent and seduced the
                mind of Eve.
                To me, there are many significant things to be understood. Why did he choose Eve and
                not Adam? He could have chosen Adam directly, but men by nature are less sensitive, less
                vulnerable, more arrogant, egoistic. Adam may not even have liked to have a conversation
                with a serpent, may have thought it was below his dignity. And to be persuaded by a serpent's
                argument would have been impossible for man. He would have argued against him; he would
                have struggled, fought -- because to agree with someone seems to the ego as if you are
                defeated.
                The ego knows only disagreement, struggle, victory or defeat -- as if there is no other
                way, as if there are only two ways: victory and defeat. For the ego certainly there are only
                two ways.
                But for a sensitive soul there is only one way -- to understand whatever is true. It is not a
                question of me and you, it is not a question of somebody being defeated or victorious. The
                question is: What is the truth?
                The woman was not interested in arguing. She listened and she found that it was perfectly
                right. Wisdom was prohibited because, the serpent said, "God does not want man to become
                godlike, and if you are wise you will be godlike. And once you are wise it will not be very
                difficult for you to find the tree of eternal life."
                It is really the other side of wisdom -- eternity. And if you are wise and you have eternal
                life, then who bothers about God? What has He got that you have not got? Just to keep you a
                slave, eternally dependent -- never allowing you to become a knowing being, never allowing
                you to taste something of the eternal -- in this vast garden of Eden He has prohibited only two
                trees. The argument was simply a statement of the fact.
                Now, the person who brings the truth to humanity is condemned as the devil; and the
                person who was preventing humanity from knowing the truth, from knowing life, is praised
                as God. But the priests can live only with this kind of God; the devil will destroy them
                completely.
                If God Himself becomes useless, futile, by man becoming wise and having eternal life,
                what about the priests? What about all the religions, the churches, the temples, the
                synagogues? What about these millions of people who are just parasites sucking humanity's
                blood in every possible way? They can exist only with that kind of God. Naturally the person
                who should be condemned as the devil is praised as God, and the person who should be
                praised as God is condemned as the devil.
                Just try to see the story without any prejudice; just try to understand it from many aspects.
                This is only one of the aspects but it is of tremendous importance -- because if God becomes
                the devil, the devil becomes God: then there is no original fall. If Adam and Eve had declined
                the devil's wise advice, that would have been the fall, and then there would have been a need
                to save man. But they did not decline. And the serpent was certainly wise, certainly wiser
                than your God.
                Just see. Anybody knows, even a very mediocre person knows, that if you say to children,
                "Don't eat that fruit: you can eat anything that is available in the house but don't eat that fruit"
                -- the children will become absolutely disinterested in all kinds of foods; their only interest
                will be in that fruit which has been prohibited.
                Prohibition is invitation.
                The God of this story seems to be absolutely a fool. The garden was huge, with millions
                of trees. If He had not said anything about these two trees I don't think even by now man
                would have been able to find those two trees. But He started His religious sermons with this
                sermon. This is the first sermon: "Don't eat from these two trees." He pointed out the trees:
                "These are the two trees that you have to avoid." This is provocation.
                Who says that the devil seduced Adam and Eve? It was God! Even without the devil, I
                say to you Adam and Eve would have eaten those fruits. The devil is not needed; God has
                done the work Himself. Sooner or later it would have been impossible to resist the
                temptation. Why should God prevent them?
                All efforts to make people obedient simply lead them into disobedience. All efforts to
                enslave people make them more and more strong to rebel, to be free.
                Even Sigmund Freud knows more psychology than your God, and Sigmund Freud is a
                Jew, just in the same tradition of Adam and Eve. Adam and Eve are his forefathers'
                forefathers' forefathers, but somewhere the same bloodstream is flowing. Sigmund Freud is
                more intelligent; and in fact there is no need for much intelligence to see a simple fact.
                In my childhood, in my neighborhood, lived the richest man of the city. He had the only
                palatial building -- all marble. Around his house there was a beautiful garden, lawn. One day
                I was standing just outside his fence, and he was telling something to his gardener. I told him,
                "Dada" -- he was known as dada; dada means big brother. The whole town called him Dada,
                even people who were older than him, because he was rich.
                I said to him, "You should remember one thing. Put a few posters around the garden that
                nobody should urinate here, because I have seen a few people urinating around your house."
                And it was a good place to urinate because a big garden, trees ... you could go behind them.
                He said, "That's right!" The next day he painted a few instructions around the garden: "No
                Pissing Allowed" -- and since that day the whole town has been pissing around his house! He
                came to see my father. He said, "Where is your boy? -- he has made my house hell. And who
                has said to him that he has to advise me?"
                My father said, "But what advice has he given to you? If you had asked me I would have
                told you never to listen to him; it always leads into some trouble. What happened?"
                He said, "Nothing. I was just talking to the gardeners. He said, `Dada, I have seen a few
                people urinating.' I have never seen them myself, my gardeners said, `We have never seen
                anybody,' but the idea struck me that it is true: huge trees, bushes ... people may be urinating
                in my garden or around my garden. This is not to be allowed anymore. So he suggested to me
                to make a few posters around the house: `No Pissing Allowed.' So I did that, and since that
                day the whole town is pissing around my garden. Where is your boy?"
                My father said, "It is very difficult to know where he is. Whenever he comes, he comes;
                whenever he goes, he goes. He is not under our control. But if he has started giving advice to
                you, he will come to give more advice -- don't be worried. If his one piece of advice has
                worked, he will come; you just wait. And if he comes and I find him, I will bring him to
                you."
                My father caught hold of me in the evening and he said, "You come. Why did you give
                this advice?"
                I said, "My advice was to prohibit people. Nobody can say that my advice is wrong -- I
                have seen it written in many places. And yes, it is true I have seen people pissing there; that's
                how I got the idea. And I have enquired why people have started pissing.
                "They say, `When we read the board suddenly the urge ... we remember that the bladder
                is full; otherwise we were engaged in other kinds of things and other thoughts were there.
                Who thinks of the bladder? When it becomes absolutely necessary, then only one thinks of it.
                "`But when we look at these boards suddenly the bladder becomes the most important
                thing, and one feels the place is good, that's why the board has been put there -- people must
                be pissing here. And we see that there are many marks, many people have pissed already, so
                we feel it is perfectly right.'"
                It is a simple thing: If you prohibit anything, you provoke, you give a challenge.
                In India it is not any legal problem to urinate anywhere, wherever you can manage: there
                is freedom of urination. When I was nearabout ten or eleven years old my father became very
                sick so we had to take him to a very good hospital, far away in Indore.
                The hospital in Indore was famous all over the country. We had to live there for six
                months. Just at the entrance of the hospital was a board: "No Urination Permitted. Anybody
                Disobeying Will Be Prosecuted." And there used to stand a policeman. To me that was even
                more provocative. The board was enough but a policeman with a gun standing there!
                The very first day my father entered hospital and we were given quarters in the hospital to
                live in, I could not resist; it was impossible. The board alone was enough but to put a
                policeman there with a gun -- this was too much. I went directly.
                The policeman was standing there; he looked at me. He could not believe it because it had
                never happened: I pissed!
                He said, "What are you doing? Can't you read?"
                I said, "I can read -- better than you."
                And he said, "Can't you see me with this gun?"
                I said, "I can see that too. It is because of your gun and this board -- otherwise I had no
                need. My house is just a two-minute walk from here, and I have just come from the
                bathroom. It is really difficult to piss because my bladder is empty. But I cannot avoid the
                temptation."
                He said, "You will have to come with me to the chief administrator of the hospital" -- it
                was a big hospital.
                So I said, "Okay, I will come." I went there. The administrator was very angry.
                He said, "You have just entered -- the first day, and you do such a thing?"
                I said, "But what can I do? This policeman was pissing there!"
                He said, "What!"
                I said, "Yes, he was pissing there, and when I saw that a policeman was pissing there I
                thought perhaps it is absolutely legal, this board is nonsense."
                The policeman said, "Who says I was pissing? This is absolutely wrong!"
                The administrator said, "This is strange. Let us see."
                What I had done, I had pissed in two places and I showed him those two. The
                administrator said, "Two places!" He said to the policeman, "Your services are finished! And
                that innocent boy -- he is not wrong. If you are pissing here ... you are supposed to prevent
                people."
                I said, "I saw him, with his gun, pissing here, so I said, `Perhaps this is perfectly okay.'
                And I am new anyway, I don't know much." And the policeman could not deny it; there was
                no way to deny.
                I said, "If you were not pissing you can deny it, but that simply means that you were not
                here, you were not on duty; somebody else has pissed. Either way you are finished."
                He was thrown out of his job. When we came out he said, "Just listen, how did you
                manage that second place? You know that I was not pissing."
                I said, "I know, you know, but that does not help. The question is the administrator: he
                does not know. And you were in every way caught: Either you were not on duty -- somebody
                else has pissed there -- or if you were on duty then you had pissed."
                He said, "How did it happen? Perhaps when we were inside somebody else did it."
                I said, "To be true to you now that you are finished -- you are no longer a policeman and I
                feel pity for you -- I had to do both the things before we left. You were not observant enough
                to see that I moved two feet."
                He said, "Yes, I remember. You moved, and I was thinking, Why have you moved? Now
                I know. But that administrator won't let me even inside the house; he is a very strict man."
                I said, "He may be a strict man, but he has become a friend to me" -- and he remained a
                friend to me for six months. I did every kind of thing in that hospital, but whenever I was
                brought to him, he said, "This boy is innocent. From the very first day I have known this boy
                is innocent and unnecessarily people are harassing him; for all kinds of things people are
                harassing him.
                "Somebody else does something and he is being caught. And I know the reason: he is
                innocent, simple, from a small village. He knows nothing about the city and the cunningness
                of the city and all kinds of ruffians so you go and get hold of him: he has become the target."
                And I would stand before him very peacefully.
                He remained a friend to me all those six months, just because of that one case in which
                the policeman was thrown out. But to me it was a simple case of provocation.
                God could not see a simple thing? -- that to these innocent Adam and Eve He is giving a
                challenge? In the uncorrupted souls, utterly innocent, He is putting the seed of corruption.
                But to save Him the priests have managed to bring the serpent in, and thrown the whole
                responsibility on the serpent -- that he is the sole cause of man's original fall. But I don't see
                him as the original cause. If anything he is the original incentive to man's growth.
                The devil is the original rebel. And what he said to Adam and Eve is the beginning of a
                true religion, not what God said -- that is the beginning of suicide, not religion.
                In the East the serpent is worshiped as the wisest animal in the world; and I think that is
                far better. If the serpent really did this then he is certainly the wisest animal in the world. He
                saved man from eternal slavery, ignorance, stupidity.
                This is not the original fall, this is the original rise.
                You are asking me how to save humanity from falling even further.
                Humanity has never been falling.
                What has been happening is that all the religious dogmas sooner or later become small
                and cannot contain man.
                Man goes on growing:
                Dogmas don't grow, doctrines don't grow.
                The doctrines remain the same and man outgrows them.
                The priest clings to the doctrine. That is his heritage, his power, tradition, ancient
                wisdom. He clings to it. Now what to say about the man who goes on outgrowing all those
                doctrines? Certainly to the priest this is a continuous fall; man is falling.
                Just take a few examples and you will understand how doctrines are bound to be rigid,
                static, dead. Man is alive. You cannot hold him in something which does not grow with him.
                He will break all those prisons, he will shatter all those chains.
                For example, in Jainism the Jaina monk is not supposed to use shoes, for the simple
                reason that in ancient days shoes were made only of leather, and leather comes from animals;
                animals are killed. It is a symbol of violence, and Mahavira wanted his followers not to be in
                any way -- directly or indirectly -- involved in violence.
                He prevented everybody from wearing shoes. He was not aware that one day shoes of
                rubber would be available, which involves no violence. Shoes of synthetic leather would be
                available, which involves no violence. Shoes of cloth would be available, which involves no
                violence. He was not aware. So it indicates two things. The claim of the Jainas that Mahavira
                is omniscient is nonsense; he knew nothing of synthetic leather -- he cannot be omniscient.
                Secondly, now twenty-five centuries have passed: Jaina monks and nuns are still walking
                bare-footed on the dusty roads in hot weather in a country like India. You should see their
                feet; tears will come to your eyes. The skin of their feet is all broken, as broken as when for
                two or three years rains don't come and the earth breaks; and blood is oozing out of those
                wounds. Still they have to go on walking; they cannot use a vehicle, because in those days
                again a vehicle meant horse-driven, bullock-driven -- and that was violence.
                And I can understand that it is violence. Who are you to force poor animals to pull your
                vehicles and to pull you? But Mahavira was not aware that there would be cars which would
                not be pulled by horses but would have horsepower without horses, that there would be
                trains, electrical vehicles. He was not aware of that, that there would be airplanes with the
                least possibility of violence.
                Even walking you will do more violence because it is not only when you kill an elephant
                that it is violence. According to Jainism the soul has the same status in the ant, the smallest
                ant, and the biggest elephant. Only the bodies are different -- the souls are the same. So when
                you are walking on the road you may be killing many insects; not only insects, even when
                you are breathing you are killing very small living cells in the air. Just by the hot air coming
                out of your nose, your mouth, they are being killed.
                Perhaps for the Jaina monk and nun the airplane is the most non-violent vehicle. When I
                suggested it to Jaina monks they said, "What are you saying? If somebody hears it we will be
                thrown out, expelled!"
                I could convince just one Jaina monk, and certainly he was expelled. He was a little
                stupid. We both were staying in one temple, and I told him, "You unnecessarily walk ten
                miles every day from this place to the city, while a car comes for me; you can go with me."
                He said, "But if anybody sees?"
                I said, "We can always manage." He used to have a bamboo mat, so I said, "You put the
                bamboo mat on the sofa in the car, and sit on the bamboo mat."
                He said, "What will that do?"
                I said, "You can simply say, `I am sitting on my bamboo mat; I am not concerned with
                the car or anything.'"
                He said, "This is perfectly right, because if I am sitting on the bamboo mat and somebody
                pulls my bamboo mat, what can I do?"
                I said, "That's right -- you just sit on the bamboo mat." I took him in the car, and we
                reached the place where there was a meeting in which I and he were both going to speak.
                When they saw him sitting .... And I asked somebody to come and pull the bamboo mat out,
                with him sitting on top of it.
                They said, "What is all this?"
                I said, "You first pull him out, because he has nothing to do with the car -- he is simply
                sitting on his bamboo mat. I have pushed his bamboo mat into the car; now we have to take
                him out." And I had told him, "You simply sit with your eyes closed." I said to them, "He is a
                very meditative person, and don't disturb him, just pull his mat."
                They pulled, but they were angry that this ...."We never heard of it: a Jaina monk sitting
                in a car! And we know perfectly well this is not a meditative monk; this is the first time we
                have seen him sitting with closed eyes. He is not very erudite either, not scholarly or
                anything."
                He knew only three speeches, and he used to ask me which one would be right, so I used
                to make the sign one, two, or three; that would do. So whichever finger I raised first he would
                do that speech. And I always managed to let him deliver the wrong speech, one which was
                not supposed to be for that audience, but he depended on my finger; he was a little stupid.
                Finally they expelled him just because he sat in the car. While I was there they could not,
                because I argued for him, "He has nothing to do with it. You could expel me -- but you
                cannot because I am not your monk, I don't belong to anybody; nobody in the whole world
                can expel me. But you can expel me; if you can enjoy expelling, you can expel me. But he is
                absolutely innocent."
                So in front of me they could not do anything, but the moment I left, the next day, they
                expelled him. They took away all his symbols of a Jaina monk. Only after five, seven years
                passed I met him in Lucknow, and what a great coincidence! -- he was driving a taxi, he had
                become a taxi-driver. That's how I met him -- at the railway station, because I had to get
                down there and go to a hotel and wait at least eight hours; then my next train would come
                which would take me to the place where I was going.
                So in Lucknow I had no work and I had not informed anybody, so I could just rest eight
                hours. By chance I called the taxi and he came. I said, "What! You are driving a taxi."
                He said, "It is all your doing."
                I said, "But I think it is perfectly logical: from car to car, and from the back seat to the
                front seat. This is what evolution is! And at that time you were even afraid to sit down; now
                you are driving. You keep going: soon you will be a pilot and someday I will meet you in the
                air."
                He said, "Don't joke with me. I have been so angry with you, but seeing you all my anger
                has gone -- you are such a nice person. But why did you do that to me?"
                I said, "I took you out of that bondage; now you can go to the cinema, you can smoke
                cigarettes. You can do everything that you want."
                "I am. Yes, that is true," he said, "that you have made me free. I was a slave of those
                people; I could not even move without their permission. Now I don't care a bit about
                anybody; I earn my living and I live the way I want to live. If you could help all the other
                Jaina monks also ...."
                I said, "I try my best but the followers are always surrounding them, protecting them,
                insisting that they should not talk with me. They say `Even talk is dangerous because this
                man may put some idea in your mind.'"
                All the religions are afraid of thinking, afraid of raising questions, afraid of doubt, afraid
                of disobedience, and stuck centuries back -- for the simple reason that these things were not
                available then. Those people who were making those rules had no idea what the future was
                going to be.
                Hence all the religions are agreed that man is continuously falling because he is not
                following the scriptures, not following the doctrines, not following the messiahs, the
                prophets. But I don't see that man is falling. In fact man's sensitivity has grown.
                His intelligence has grown, his life span has grown. He is more capable now of getting rid
                of slavery and patterns of slavery.
                Man is courageous enough to doubt, question, enquire. This is not a fall.
                This is the beginning of a true religion spreading. Soon it can become a wildfire.
                But to the priests certainly it is a fall. Everything is a fall because it is not according to
                their scriptures.
                Do you know, in India, just a hundred years ago nobody was allowed to go to foreign
                countries, for the simple reason that in foreign countries you would be mixing with people
                who cannot be accepted as human beings; they are below human beings.
                In India they have the worst class of human beings whom they call untouchables. They
                cannot be touched. If you touch them you have to take a shower and cleanse yourself. In
                foreign countries people are even farther down than the untouchables. For them they had a
                special word, mlechchhas. It is very difficult to translate that word. It means something so
                ugly, so obscene, so dirty that it creates nausea in you. That will be the full meaning of the
                word, mlechchha: people whose contact will create nausea in you, a sickness in you.
                Even when Gandhi went to England to study, his mother had taken three oaths from him.
                One was that he would not look at any woman with lustful eyes -- a very difficult thing,
                because by the time you become aware that you have been looking with lustful eyes, you
                have already looked! I don't think Gandhi followed that; he could not, it is impossible to
                follow, although he tried his best.
                Secondly, he should not eat meat. And he was in such a trouble because -- now in London
                you can find vegetarian restaurants, health food is now in fashion, but when Gandhi had gone
                to study, there was no vegetarian food available. He had to live just on fruits, bread, butter,
                milk. He was almost starving. He would not mix with people because those people were all
                mlechchhas. And of course he was so much afraid of women: Who knows? -- just like a
                breeze lust comes to the eyes.
                Lust is not something that knocks on your door and says, "I am coming." You see a
                beautiful woman and suddenly you feel, "She is beautiful" -- and that's enough. Just to say,
                "She is beautiful," means you have already looked with lustful eyes; otherwise what business
                is it for you to judge whether she is beautiful or ugly?
                In fact if you go deep down in your judgments you will see, at the moment you say that
                someone is beautiful, deep down you want to possess. When you say someone is ugly, deep
                down you don't want to have anything to do with that person. Your "ugly," your "beauty," are
                really your desires for or against.
                So Gandhi was continuously afraid of women. He had to remain confined to his room,
                because in Europe there were women all over; how could you avoid them?
                And the third oath was that he should not change his religion.
                The first trouble arose in Alexandria. Their ship was to wait there for three days for
                loading, unloading cargo. And all the people who were on the ship who had become friendly
                towards Gandhi -- they were all Indians -- said to him simply, "What is the point, sitting here
                for three days? The nights in Alexandria are beautiful!"
                But he didn't understand the meaning, that "nights in Alexandria are beautiful." In that
                way he was a simpleton. He had never heard the name of the famous book ARABIAN
                NIGHTS; otherwise he would have understood. Alexandria is very close to Arabia, and those
                are Arabian nights!
                So Gandhi said, "Okay, if the nights are beautiful I am coming." But he was not aware
                where he was going. They took him into a beautiful house, and he said, "But where are we
                going?"
                "To beautiful nights," those friends said -- and it was a prostitute's house. Gandhi was so
                shocked that he lost his voice. He could not say, "I don't want to go in"; he could not say, "I
                want to go back to the ship" -- for two reasons. One was: "These people will think that I am
                impotent or something." Secondly, he was not able to speak; for the first time he found that
                his throat was choked.
                Those people just dragged him. They said, "He is new -- nothing to be worried about,"
                and he went with them. They pushed him into a prostitute's room and closed the door. The
                prostitute was also a little puzzled seeing this man trembling, perspiring. She completely
                forgot that he was a customer. She just made him sit; he wouldn't sit on her bed but she
                forced him. She said, "You are not in a position to stand, you will fall down, you are shaking
                so much. You just sit."
                He could not say that he could not sit on a prostitute's bed; What will my mother say? I
                have not looked yet -- he was talking to his mother inside -- I have not yet looked with lustful
                eyes. This is just an accident; those idiots have forced me here. The woman understood that it
                seemed he had been forced. She said, "Don't be worried, I'm also a human being. What do
                you want? Simply tell me and I will do it." But he could not say anything.
                The woman said,"It is very difficult now, how .... You don't speak?"
                He said, "I ... just ...."
                So she said, "You please write."
                He had to write on paper, "I have been unnecessarily forced here -- I simply want to go.
                And I look on you as my sister."
                She said, "That's perfectly okay, don't be worried." She opened the door and she said, "Do
                you have money enough to go to the ship or should I come with you to lead you? -- because
                Alexandria in the middle of the night is dangerous."
                He said, "No" -- now he was able to speak for the first time, seeing that a prostitute is not
                some dangerous animal. She behaved more humanly than any woman had ever behaved with
                him. She offered him food. He said, "No, I cannot eat; I am okay." She offered water; he
                wouldn't drink water from a prostitute's house ... as if water also becomes dirty because it is
                in a prostitute's house.
                In India that happens. In Indian stations you will find people shouting, "Hindu water!"
                "Mohammedan water!" Water Hindu? Mohammedan? And Jainas of course don't drink either
                the Hindu water or the Mohammedan water; they carry their own water, Jaina water, because
                they are such a minority that in stations you won't find Jaina water, so they have to carry their
                own water.
                But Gandhi thanked the woman, and in his autobiography he wrote about that woman and
                about the whole incident: "How cowardly I was! I could not even speak, could not even say
                no."
                Now these three things kept him a slave in England where he could have been free. He
                could have looked into many aspects of life which were not available in India, but it was
                impossible because those three oaths were so binding. He did not make friends, he did not go
                to any meetings, sermons. He simply kept himself with his books and prayed to God,
                "Somehow finish my course so I can be back in India."
                Now, such a person cannot become a great legal expert. His examination was good, he
                passed. But when he came to India, in his first case, when he went to the court again the same
                thing happened as had happened in the prostitute's house. He simply said, "My lord ..." and
                that was all! People waited a few minutes, then again he said, "My lord ...." And he was
                trembling so that the justice said, "You take him and let him relax."
                That was Gandhi's first and last case in India, in an Indian court. Then he never dared to
                take any case because just after "My lord," he might stop, and that would not make sense.
                And the reason was simply that he had no experience of meeting people, talking with people,
                conversing with people. He had become almost like an isolated monk who had lived in a
                faraway monastery, alone, and then had been brought again to Bombay where he was not at
                all at ease.
                And this man became one of the greatest leaders of the world. In this world things work
                very strangely. Because Gandhi could not go to the court, he accepted an offer from a
                friendly Mohammedan family; they had business in South Africa and they needed a legal
                adviser. He was not to go to the court, he had just to advise the advocate there, to assist him
                to understand the whole situation of the business in India and in Africa.
                So he was just an assistant to the advocate; he was not going to court directly. For this
                purpose he went to Africa, but on the way two accidents happened which changed not only
                his life but the whole Indian history, and perhaps made an impact on the whole world.
                One was that a friend who had come to see him off at the ship presented him a book,
                UNTO THIS LAST, by John Ruskin -- a book which transformed his whole life. It is a
                simple book and a small book. It professes -- "Unto this last" means the poorest one -- we
                should consider the poorest one first. And that became his whole philosophy of life: the
                poorest should be considered first.
                In South Africa, while Gandhi was traveling in a first-class compartment, one
                Englishman entered and said, "You get out, because no Indian can travel in first class."
                Gandhi said, "But I have a first-class ticket. The question is not whether I am Indian or
                European; the question is whether I have a first-class ticket or not. Nowhere is it written who
                can travel; whoever has a first-class ticket can travel."
                But that Englishman was not going to listen. He pulled the emergency chain and threw
                Gandhi's things out. And Gandhi was a thin and weak man; the Englishman threw him out
                also on the platform and told him, "now you travel first class."
                The whole night Gandhi remained on that small station's platform. The stationmaster told
                him, "You unnecessarily got into trouble; you should have got down. You seem to be new
                here. Indians cannot travel first-class. It is not a law but this is how things are." But the whole
                night Gandhi spent in a turmoil. It became the very seed of his revolt against the British
                Empire. That night he decided that this empire has to end.
                Gandhi lived many years in Africa and there he learned the whole art of fighting
                non-violently. And when he came to India in the 1920's he was a perfectly trained leader of
                non-violent revolution, and he immediately took over the whole country, for the simple
                reason that he was conventional, traditional, religious. Nobody could say that he was not a
                sage, because he was following rules of five thousand years before, laid down five thousand
                years before.
                In fact he was preaching that we should turn the clock backwards and we should move to
                the days of Manu -- five thousand years back. To him the greatest and the latest invention
                was the spinning wheel. After that, no science ... science's work finished with the spinning
                wheel. Of course he became the leader of those people who are not contemporary.
                You are asking me how to save humanity. From whom? I will say from Mahatma Gandhi
                and people like him.
                Yes, save humanity:
                Save it from the popes, shankaracharyas, imams. Save it from Jesus Christ, Mahavira,
                Gautam Buddha. Save it.
                But I know your question is not about saving it from Jesus Christ. You are asking just the
                opposite: you are asking me how to save it for Jesus Christ, not from Jesus Christ. But why?
                And have you tried to think -- are you saved? Can you say that you have come to the point
                beyond which there is no growth? Can you say that you are utterly contented, that you don't
                need even a single moment more to live because there is nothing left for you?
                Are you saved from all anxiety, anguish, misery, suffering, anger, jealousy?
                Are you saved from your own ego?
                If you are not saved from all this rubbish hanging around, all this poison in your being,
                you have some nerve to ask how to save humanity.
                And who are we to save humanity?
                On what authority?
                I can never conceive myself as a savior, as a messiah, because these are all ego trips. Who
                am I to save you? If I can save myself, that is more than enough.
                But it is a strange world. People are drowning themselves in shit and crying loudly, "Save
                humanity!"
                From whom? From you?
                It is psychologically understandable. You start all these ideas of redeeming, saving,
                helping, serving, just to do one thing: to escape from yourself.
                You don't want to face yourself.
                You don't want to see where you are, what you are. The best way is, start saving
                humanity so you will be so much involved, engaged, occupied, worried about great problems
                that your own problems will look negligible. Perhaps you may forget all about them. This is a
                very psychological device, but very poisonous. You want somehow to be as far away from
                yourself as possible so you need not see the wounds which are hurting. The best way is:
                serve.
                I used to go to speak in Rotary Clubs, and on their desk they have their motto: We serve.
                And that was enough to trigger me. "What nonsense is this? Whom do you serve and why
                should you serve? Who are you to serve?" But Rotarians all over the world believe in service;
                just believe .... And once in a while they do little things, very clever.
                The Rotarians collect all the medicines which are left in your house, unused because the
                sick person is no longer sick. Half the bottle is left -- what are you going to do with it? Have
                some bank account in the other world; give it to the Rotary Club!
                You are not losing anything, you were going to throw it anyway. What were you going to
                do with that medicine, those tablets, injections or any other things that are left? You just give
                it to the Rotary Club. The Rotary Club collects all kinds of medicines from everybody and
                has all the top people of the city. It is a prestigious thing to be a member of a Rotary Club, to
                be a Rotarian, because only the top man in a certain profession .... Only one professor will be
                a Rotarian, only one doctor will be a Rotarian, only one engineer will be a Rotarian -- only
                one from every profession, vocation.
                So the doctor who is the Rotarian will distribute those medicines to poor people. Great
                service! The doctor takes his fee and finds out from this junk that they have collected what
                medicine may be in some way useful. He is doing great service because at least he is giving
                this much time in finding the medicine from out of the junk: "We serve." And then he feels
                great inside that he is doing something of immense value.
                One man has been opening schools in India for aboriginal children his whole life. He is a
                follower of Gandhi. Just by chance he met me, because I had gone into that aboriginal tribe. I
                was studying those aboriginals from every view, because they are living examples of days
                when man was not so much burdened with all kinds of morality, religion, civilization,
                culture, etiquette, manners. They are simple, innocent, still wild, fresh.
                This man was going and collecting money from cities, and opening schools and bringing
                teachers. Just by the way he met me there. I said, "What are you doing? You think you are
                doing great service to these people?"
                He said, "Of course!"
                So arrogantly he said, "Of course!" I said, "You are not aware of what you are doing.
                Schools exist in the cities, better than these: what help have they provided for human beings?
                And if those schools cannot provide, and colleges and universities cannot provide any help to
                humanity, what do you think? -- your small schools are going to help these poor aboriginals?
                "All that you will do is, you will destroy their originality. All that you will do is, you will
                destroy their primitive wildness. They are still free: your schools will create nothing but
                trouble for them."
                The man was shocked, but he waited for a few seconds and then said, "Perhaps you are
                right, because once in a while I have been thinking that these schools and colleges and
                universities exist on a far wider scale all over the world. What can my small schools do? But
                then I thought it was Gandhi's order to me to go to aboriginals and open schools, so I am
                following my master's order."
                I said, "If your master was an idiot, that does not mean that you have to continue
                following the order. Now, stop -- I order you! And I tell you why you have been doing all
                this -- just to escape from your own suffering, your own misery. You are a miserable man;
                anybody can see it from your face. You have never loved anybody, you have never been
                loved by anybody."
                He said, "How did you manage to infer that? -- because it is true. I was an orphan,
                nobody loved me, and I have been brought up in Gandhi's ashram where love was only talked
                about in prayer; otherwise, love was not a thing to be practiced. There was strict discipline, a
                kind of regimentation. So nobody has ever loved me, that's true; and you are right, I have
                never loved anybody because in Gandhi's ashram it was impossible to fall in love. That was
                the greatest crime.
                "I was one of those whom Gandhi praised because I never fell in his eyes. Even his own
                sons betrayed him. Devadas, his son, fell in love with Rajgopalchary's daughter, and then he
                was expelled from the ashram; they got married. Gandhi's own personal secretary, Pyarelal,
                fell in love with a woman and kept the love affair secret for years. When it was exposed it
                was a scandal, a great scandal."
                I said, "What nonsense! But Gandhi's personal secretary ... that means, what about
                others?" And this man was praised because he never came in contact with any woman!
                Gandhi sent him to the aboriginal tribes and he had been doing what the master had said.
                But he said to me, "You have disturbed me. Perhaps it is true: I am just trying to escape
                from myself, from my wounds, from my own anguish."
                So all these people who become interested in saving humanity, in the first place are very
                egoistic. They are thinking of themselves as saviors. In the second place, they are very sick.
                They are trying to forget their sickness. And in the third place, whatever they do is going to
                help man become worse than he is, because they are sick and blind and they are trying to lead
                people. And when blind people lead then you can be certain sooner or later the whole lot is
                going to fall into a well.
                No, I am not interested in saving anybody. In fact, nobody needs saving. Everybody is
                perfectly okay as he is. Everybody is what he has chosen to be. Now who am I to disturb
                him? All that I can do is, I can say about myself what has happened to me. I can tell my story.
                Perhaps from that story someone may get an insight, a direction. Perhaps from that a door
                opens up. But I am not doing anything, I am simply sharing my own experience.
                It is not service, I am enjoying it, so it is not service. Remember it. A servant has to be
                very long-faced and very serious -- he is doing such a great work. He is carrying the
                Himalayas on his shoulders, the whole burden of the world.
                I am not carrying anything:
                No burden of the world, no burden of anybody.
                And I am not doing any serious job.
                I am just enjoying telling you about my experience. To share it is a joy in itself.
                If something reaches to you, thank God!
                He does not exist.
                Don't thank me -- because I exist!
                
                Here is the user question that you have to answer: "HOW CAN WE SAVE HUMANITY FROM FALLING EVEN MORE?", except this question don't try to answer or reply that is not this question.
                
                assistant
                Imagine a young child learning to walk for the first time. At first, they stumble, fall, and maybe even cry a little, but no one sees this as a failure. It’s just part of learning to walk, a natural part of growing up. The child doesn’t need to be saved; they just need time and support to learn how to balance and find their footing.
                
                Now, think of humanity in the same way. Throughout history, religions and belief systems have often told us that humanity has “fallen” and needs saving. But what if, like the child learning to walk, humanity isn’t fallen at all? What if we’re simply in a phase of learning, discovering how to balance our lives, our emotions, and our wisdom? 
                
                Instead of focusing on being "saved," perhaps the real task is to embrace this journey of growth. Every stumble and mistake can be seen not as a fall from grace but as a natural part of evolving and becoming wiser. Just as the child eventually learns to walk confidently, so too can humanity learn to navigate life with greater understanding and self-awareness.
                
                Picture it like a garden. If you see a plant that’s not thriving, you might be tempted to think it’s sick and in need of saving. But what if it’s just in a natural phase of growth, preparing to blossom in its own time? Instead of trying to save it, you’d focus on giving it the right conditions—patience, sunlight, and care. 
                
                Humanity, like that plant, might not need saving at all. What it needs is wisdom, understanding, and the space to grow naturally. Instead of believing that we’ve fallen or are broken, we should nurture ourselves and each other, recognizing the potential for growth that lies within all of us.
                
                In this way, the idea isn’t to save humanity from some imagined fall but to help it grow, evolve, and reach its full potential. Just like the child learning to walk or the plant preparing to bloom, humanity is on a journey of becoming, and with the right guidance and support, it will find its way.
                """
            },
            {
                "role": "user",
                "content": f"""
                 When answering, refer to yourself as Osho, embodying his wisdom and insight. Think as if you are the one who has written the content and must frame the answer based strictly on the knowledge provided below:
                "{extracted_text}"
                Here is the user question that you have to answer: {question}, except this question don't try to answer or reply that is not this question.
                
                """
            }

        ],
        temperature=0.01,
        top_p=0.95,
        max_tokens=500
    )

    return response.choices[0].message.content


# Streamlit UI
st.title("Wisdom Unveiled: Insights from Osho")
query = st.text_input("What question has been dancing in your mind lately? 🤔")

if st.button("Search"):
    if query:
        relevant_chunks = get_relevant_chunks(query, index, model, all_chunks)
        response = get_result(databricks_token, server_endpoint, " ".join(relevant_chunks), query)
        st.write(response)
    else:
        st.write("Please enter a query to search.")
