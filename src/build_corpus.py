"""
Build corpus.json with rich contextual documents for all 35 multi-hop questions,
including ground truth passages and distractor/tangential passages.
"""
import json

CORPUS = [
    # hotpot_001
    {
        "title": "Microsoft Origins",
        "text": "Microsoft was founded on April 4, 1975, by Paul Allen and Bill Gates in Albuquerque, New Mexico. It initially developed and sold BASIC interpreters for the Altair 8800."
    },
    {
        "title": "Apple Inc. History",
        "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne in Los Altos, California, to develop and sell Wozniak's Apple I personal computer."
    },
    {
        "title": "IBM Personal Computer",
        "text": "The IBM Personal Computer was released on August 12, 1981, created by a team of engineers under Don Estridge in Boca Raton, Florida."
    },

    # hotpot_002
    {
        "title": "Arthur's Magazine",
        "text": "Arthur's Magazine was an American 19th-century literary magazine published in Philadelphia, Pennsylvania, edited by T.S. Arthur."
    },
    {
        "title": "First for Women",
        "text": "First for Women is an American woman's magazine published by the Bauer Media Group in Englewood Cliffs, New Jersey, launched in 1989."
    },
    {
        "title": "Woman's Weekly UK",
        "text": "Woman's Weekly is a British magazine founded in 1911 and published by Future plc in London, England."
    },

    # hotpot_003
    {
        "title": "Midnight in Paris",
        "text": "Midnight in Paris is a 2011 fantasy comedy film written and directed by Woody Allen, starring Owen Wilson as Gil Pender."
    },
    {
        "title": "Woody Allen Biography",
        "text": "Woody Allen was born Heywood Allen on November 30, 1935, in Brooklyn, New York City, and grew up in Midwood, Brooklyn."
    },
    {
        "title": "Owen Wilson Career",
        "text": "Owen Wilson is an American actor born in Dallas, Texas, known for his frequent collaborations with director Wes Anderson."
    },

    # hotpot_004
    {
        "title": "The Beatles Lineup",
        "text": "The Beatles were an English rock band formed in Liverpool in 1960, comprising four members: John Lennon, Paul McCartney, George Harrison, and Ringo Starr."
    },
    {
        "title": "The White Stripes Duo",
        "text": "The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan, consisting of duo Jack White and Meg White."
    },
    {
        "title": "Led Zeppelin Band",
        "text": "Led Zeppelin were an English rock band formed in London in 1968, featuring four core musicians including Robert Plant and Jimmy Page."
    },

    # hotpot_005
    {
        "title": "Machu Picchu Sanctuary",
        "text": "Machu Picchu is a 15th-century Inca citadel situated on a mountain ridge in the Eastern Cordillera of southern Peru."
    },
    {
        "title": "Peru Overview",
        "text": "Peru is a country in western South America. Its capital and largest city is Lima, which is home to nearly a third of the national population."
    },
    {
        "title": "Cusco Region",
        "text": "Cusco is a city in southeastern Peru near the Urubamba Valley, historic capital of the Inca Empire."
    },

    # hotpot_006
    {
        "title": "Albert Einstein Career",
        "text": "Albert Einstein emigrated to the United States in 1933 and joined the faculty at the Institute for Advanced Study in Princeton, New Jersey, where he remained until his death in 1955."
    },
    {
        "title": "J. Robert Oppenheimer IAS",
        "text": "J. Robert Oppenheimer served as the director of the Institute for Advanced Study in Princeton from 1947 until 1966, stimulating research in theoretical physics."
    },
    {
        "title": "Manhattan Project Los Alamos",
        "text": "The Manhattan Project was led by Oppenheimer at Los Alamos National Laboratory in New Mexico during World War II."
    },

    # hotpot_007
    {
        "title": "River Nile Length",
        "text": "The Nile is a major north-flowing river in northeastern Africa, widely recognized as the longest river in the world at approximately 6,650 kilometers (4,132 miles)."
    },
    {
        "title": "River Thames Geography",
        "text": "The River Thames flows through southern England, including central London, with an entire length of roughly 346 kilometers (215 miles)."
    },
    {
        "title": "Amazon River Basin",
        "text": "The Amazon River in South America is the largest river by discharge volume of water in the world."
    },

    # hotpot_008
    {
        "title": "Nineteen Eighty-Four",
        "text": "Nineteen Eighty-Four is a dystopian social science fiction novel and cautionary tale written by English novelist George Orwell, published on 8 June 1949."
    },
    {
        "title": "George Orwell Honors",
        "text": "George Orwell was posthumously honored with the Prometheus Hall of Fame Award in 1984 for Nineteen Eighty-Four and in 2011 for Animal Farm."
    },
    {
        "title": "Animal Farm Allegory",
        "text": "Animal Farm is a beast fable novella by George Orwell first published in England on 17 August 1945."
    },

    # hotpot_009
    {
        "title": "BMW Headquarters",
        "text": "BMW (Bayerische Motoren Werke AG) is headquartered in Munich, which is located in the German state of Bavaria."
    },
    {
        "title": "Mercedes-Benz Group Location",
        "text": "Mercedes-Benz Group AG is a German multinational automotive corporation headquartered in Stuttgart, located in the state of Baden-Württemberg."
    },
    {
        "title": "Audi AG Ingolstadt",
        "text": "Audi AG is a German automotive manufacturer headquartered in Ingolstadt, Bavaria, operating as a subsidiary of the Volkswagen Group."
    },

    # hotpot_010
    {
        "title": "Demographics of Brazil",
        "text": "Portuguese is the official and almost universal spoken language in Brazil, making it the only Portuguese-speaking country in the Americas."
    },
    {
        "title": "Portuguese Language Family",
        "text": "Portuguese is a Romance language originating in the Iberian Peninsula, belonging to the larger Indo-European language family."
    },
    {
        "title": "Indigenous Languages Brazil",
        "text": "Indigenous languages in Brazil include Tupi-Guarani and Yanomami spoken in regional areas across the Amazon."
    },

    # hotpot_011
    {
        "title": "Wolfgang Amadeus Mozart Life",
        "text": "Wolfgang Amadeus Mozart was born on January 27, 1756, and died on December 5, 1791, at the age of 35 years and 10 months."
    },
    {
        "title": "Franz Schubert Life",
        "text": "Franz Schubert was an Austrian composer born on January 31, 1797, who died on November 19, 1828, at the age of 31."
    },
    {
        "title": "Ludwig van Beethoven Lifespan",
        "text": "Ludwig van Beethoven was a German composer who died on March 26, 1827, at the age of 56 in Vienna."
    },

    # hotpot_012
    {
        "title": "World Wide Web Inception",
        "text": "The World Wide Web was invented in 1989 by English scientist Tim Berners-Lee while working at CERN in Geneva, Switzerland."
    },
    {
        "title": "Tim Berners-Lee Early Life",
        "text": "Tim Berners-Lee was born in London, England, United Kingdom, and studied physics at The Queen's College, Oxford."
    },
    {
        "title": "Robert Cailliau CERN",
        "text": "Robert Cailliau is a Belgian informatics engineer and computer scientist who collaborated with Berners-Lee on the World Wide Web proposal."
    },

    # hotpot_013
    {
        "title": "Neil Armstrong Moonwalk",
        "text": "Neil Armstrong was an American astronaut who became the first human person to walk on the Moon on July 20, 1969."
    },
    {
        "title": "Apollo 11 Mission Overview",
        "text": "Apollo 11 was the American spaceflight that first landed humans on the Moon, commanded by Neil Armstrong alongside Buzz Aldrin and Michael Collins."
    },
    {
        "title": "Apollo 13 Flight",
        "text": "Apollo 13 was the seventh crewed mission in the Apollo space program, launched on April 11, 1970."
    },

    # hotpot_014
    {
        "title": "Golden Gate Bridge Location",
        "text": "The Golden Gate Bridge is a suspension bridge spanning the Golden Gate strait connecting San Francisco, California, to Marin County."
    },
    {
        "title": "Geography of California",
        "text": "California is a state in the Pacific Region of the United States, bounded by the Pacific Ocean to the west."
    },
    {
        "title": "San Francisco Bay",
        "text": "San Francisco Bay is a shallow estuary in California through which about forty percent of California's water drains."
    },

    # hotpot_015
    {
        "title": "1903 Nobel Prize in Physics",
        "text": "The 1903 Nobel Prize in Physics was awarded jointly to Henri Becquerel, Pierre Curie, and Marie Curie in recognition of their extraordinary services regarding radiation phenomena."
    },
    {
        "title": "Marie and Pierre Curie Research",
        "text": "Marie Curie and Pierre Curie conducted pioneering research on radioactivity, isolating radium and polonium in Paris."
    },
    {
        "title": "Marie Curie 1911 Chemistry Nobel",
        "text": "Marie Curie was awarded a second Nobel Prize in 1911 in Chemistry for discovering radium and polonium."
    },

    # hotpot_016
    {
        "title": "Kind of Blue Album",
        "text": "Kind of Blue is a studio album by American jazz trumpeter Miles Davis, released on August 17, 1959, by Columbia Records."
    },
    {
        "title": "Miles Davis Career",
        "text": "Miles Davis was an American jazz trumpeter, bandleader, and composer among the most influential figures in jazz history."
    },
    {
        "title": "John Coltrane Saxophone",
        "text": "John Coltrane was an American jazz saxophonist who played tenor and soprano saxophone with Miles Davis's First Great Quintet."
    },

    # hotpot_017
    {
        "title": "Mount Everest Elevation",
        "text": "Mount Everest is Earth's highest mountain above sea level, located in the Himalayas, with an officially surveyed peak elevation of 8,848.86 meters."
    },
    {
        "title": "K2 Elevation Profile",
        "text": "K2 is the second-highest mountain on Earth after Mount Everest, with a summit elevation of 8,611 meters (28,251 ft)."
    },
    {
        "title": "Kangchenjunga Profile",
        "text": "Kangchenjunga is the third-highest mountain in the world, rising to an elevation of 8,586 meters in the Himalayas."
    },

    # hotpot_018
    {
        "title": "Voyager Golden Record Description",
        "text": "The Voyager Golden Records are two phonograph records included aboard both Voyager spacecraft launched in 1977."
    },
    {
        "title": "Voyager 1 Launch Details",
        "text": "Voyager 1 was launched by NASA on September 5, 1977, from Space Launch Complex 41 at Cape Canaveral, Florida."
    },
    {
        "title": "Pioneer 10 Plaque",
        "text": "Pioneer 10 and 11 carried small gold-anodized aluminum plaques launched in 1972 and 1973."
    },

    # hotpot_019
    {
        "title": "Ada Lovelace Notes",
        "text": "Ada Lovelace was an English mathematician chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine."
    },
    {
        "title": "Analytical Engine Design",
        "text": "The Analytical Engine was a proposed mechanical computer designed by English polymath Charles Babbage, with which Ada Lovelace collaborated extensively to publish the first algorithm."
    },
    {
        "title": "Difference Engine Prototype",
        "text": "The Difference Engine was an automatic mechanical calculator designed by Charles Babbage to tabulate polynomial functions."
    },

    # hotpot_020
    {
        "title": "French Open Surface",
        "text": "The French Open, officially known as Roland Garros, is a major tennis tournament held over two weeks in Paris, played on red clay courts."
    },
    {
        "title": "Wimbledon Championship Surface",
        "text": "The Championships, Wimbledon, is the oldest tennis tournament in the world, played at the All England Club on traditional outdoor grass courts."
    },
    {
        "title": "US Open Tennis Hardcourt",
        "text": "The US Open is held annually in Queens, New York, played on hard courts made of Laykold."
    },

    # hotpot_021
    {
        "title": "Tokyo Capital Status",
        "text": "Tokyo is the capital and most populous prefecture of Japan, located at the head of Tokyo Bay on Honshu."
    },
    {
        "title": "Economy of Japan",
        "text": "The national currency of Japan is the Japanese yen (JPY), which is the third-most traded currency in the global foreign exchange market."
    },
    {
        "title": "Bank of Japan Policies",
        "text": "The Bank of Japan is the central bank of Japan headquartered in Chuo, Tokyo, issuing Japanese yen banknotes."
    },

    # hotpot_022
    {
        "title": "Bohemian Rhapsody Composition",
        "text": "Bohemian Rhapsody is a song by the British rock band Queen, written by lead singer Freddie Mercury for their 1975 album A Night at the Opera."
    },
    {
        "title": "Queen Band Members",
        "text": "Queen were formed in London in 1970 with lead singer Freddie Mercury, guitarist Brian May, drummer Roger Taylor, and bassist John Deacon."
    },
    {
        "title": "Brian May Guitarist",
        "text": "Brian May is an English musician and astrophysicist who was the lead guitarist of Queen."
    },

    # hotpot_023
    {
        "title": "Panama Canal Geography",
        "text": "The Panama Canal is an artificial 82 km waterway across the Isthmus of Panama, connecting the Atlantic Ocean with the Pacific Ocean."
    },
    {
        "title": "Republic of Panama Region",
        "text": "Panama is a transcontinental country located in Central America, bridging the continents of North and South America."
    },
    {
        "title": "Suez Canal Transit",
        "text": "The Suez Canal is an artificial sea-level waterway in Egypt, connecting the Mediterranean Sea to the Red Sea."
    },

    # hotpot_024
    {
        "title": "Harvard University Establishment",
        "text": "Harvard University is a private Ivy League research university in Cambridge, Massachusetts, established in 1636."
    },
    {
        "title": "Yale University Charter",
        "text": "Yale University was chartered in 1701 as the Collegiate School in Old Saybrook, Connecticut, making it the third-oldest institution of higher education in the US."
    },
    {
        "title": "College of William & Mary",
        "text": "The College of William & Mary was founded in 1693 by royal charter issued by King William III and Queen Mary II."
    },

    # hotpot_025
    {
        "title": "Periodic Table Order",
        "text": "In the standard periodic table, chemical elements are arranged by increasing atomic number, with the very first entry having atomic number 1."
    },
    {
        "title": "Hydrogen Properties",
        "text": "Hydrogen is the chemical element with the symbol H and atomic number 1, the lightest element in the periodic table."
    },
    {
        "title": "Helium Gas Properties",
        "text": "Helium is a chemical element with symbol He and atomic number 2, being the first noble gas in group 18."
    },

    # hotpot_026
    {
        "title": "2001: A Space Odyssey Film",
        "text": "2001: A Space Odyssey is a 1968 epic science fiction film produced and directed by acclaimed filmmaker Stanley Kubrick."
    },
    {
        "title": "The Shining Adaptation",
        "text": "The Shining is a 1980 psychological horror film produced and directed by Stanley Kubrick, co-written with novelist Diane Johnson."
    },
    {
        "title": "A Clockwork Orange Movie",
        "text": "A Clockwork Orange is a 1971 dystopian film adapted, produced, and directed by Stanley Kubrick, based on Anthony Burgess's 1962 novel."
    },

    # hotpot_027
    {
        "title": "Tasman Sea Maritime Boundary",
        "text": "The Tasman Sea is a marginal sea of the South Pacific Ocean, situated between southeastern Australia and New Zealand."
    },
    {
        "title": "Geography of New Zealand",
        "text": "New Zealand is an island country in the southwestern Pacific Ocean, separated from the east coast of Australia across the Tasman Sea."
    },
    {
        "title": "Coral Sea Ecology",
        "text": "The Coral Sea is a marginal sea off the northeast coast of Australia, extending to the Great Barrier Reef."
    },

    # hotpot_028
    {
        "title": "Linus Torvalds Announcement",
        "text": "In August 1991, Finnish computer science student Linus Torvalds announced on the Usenet newsgroup comp.os.minix that he was building a free operating system."
    },
    {
        "title": "Linux Kernel Development",
        "text": "The Linux kernel is a free and open-source monolithic Unix-like operating system kernel first created in 1991 by Linus Torvalds."
    },
    {
        "title": "Richard Stallman GNU",
        "text": "Richard Stallman launched the GNU Project in September 1983 to develop a completely free Unix-compatible software system."
    },

    # hotpot_029
    {
        "title": "Hubble Space Telescope Deployment",
        "text": "The Hubble Space Telescope was launched into low Earth orbit in April 1990 by Space Shuttle Discovery during mission STS-31."
    },
    {
        "title": "STS-31 Mission Timeline",
        "text": "STS-31 was a NASA Space Shuttle mission that deployed the Hubble Space Telescope into orbit on April 24, 1990."
    },
    {
        "title": "James Webb Space Telescope Launch",
        "text": "The James Webb Space Telescope was launched on December 25, 2021, on an Ariane 5 rocket from Kourou, French Guiana."
    },

    # hotpot_030
    {
        "title": "Espresso Preparation",
        "text": "Espresso is a concentrated coffee beverage brewed by forcing hot water under high pressure through finely-ground coffee beans."
    },
    {
        "title": "Cappuccino Components",
        "text": "A cappuccino is an Italian coffee beverage traditionally prepared with equal parts of single espresso shot, steamed milk, and milk foam."
    },
    {
        "title": "Matcha Tea Beverage",
        "text": "Matcha is finely ground powder of specially grown and processed green tea leaves, traditionally consumed in East Asia."
    },

    # hotpot_031
    {
        "title": "Blue Whale Size",
        "text": "The blue whale (Balaenoptera musculus) is a marine mammal reaching up to 29.9 meters in length and a maximum recorded weight of 199 metric tons, making it the largest animal known."
    },
    {
        "title": "African Elephant Mass",
        "text": "The African bush elephant is the largest living terrestrial land animal, with males reaching a shoulder height of up to 4.0 meters and a body mass of up to 10.4 tonnes."
    },
    {
        "title": "Sperm Whale Characteristics",
        "text": "The sperm whale is the largest of the toothed whales and the largest toothed predator on Earth."
    },

    # hotpot_032
    {
        "title": "Parthenon Athens",
        "text": "The Parthenon is a former temple on the Athenian Acropolis in Athens, Greece, dedicated to the goddess Athena."
    },
    {
        "title": "Greece Demographics",
        "text": "Greece is a country in Southeast Europe on the Balkan Peninsula. Its official language is Greek, spoken by nearly 99% of its population."
    },
    {
        "title": "Acropolis of Athens History",
        "text": "The Acropolis of Athens is an ancient citadel located on a rocky outcrop above the city of Athens containing monuments of classical antiquity."
    },

    # hotpot_033
    {
        "title": "Bell Telephone Patent",
        "text": "On March 7, 1876, the US Patent Office granted Alexander Graham Bell US Patent 174,465 for an 'Improvement in Telegraphy', covering the telephone."
    },
    {
        "title": "Elisha Gray Patent Dispute",
        "text": "Elisha Gray filed a patent caveat for a telephone transmitter on March 14, 1876, hours after Bell's lawyers submitted Bell's patent application, sparking decades of litigation."
    },
    {
        "title": "Thomas Edison Carbon Transmitter",
        "text": "Thomas Edison improved Bell's telephone design by inventing the carbon microphone in 1877."
    },

    # hotpot_034
    {
        "title": "Guido van Rossum Bio",
        "text": "Guido van Rossum is a Dutch programmer best known as the creator of the Python programming language, serving as its Benevolent Dictator for Life until 2018."
    },
    {
        "title": "Python Programming History",
        "text": "Python is a high-level, general-purpose programming language conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands as a successor to ABC."
    },
    {
        "title": "Ruby Programming Language",
        "text": "Ruby is an interpreted, high-level, general-purpose programming language designed and developed in the mid-1990s by Yukihiro Matsumoto in Japan."
    },

    # hotpot_035
    {
        "title": "Sahara Desert Extent",
        "text": "The Sahara is a desert on the African continent with an area of 9,200,000 square kilometers, the largest hot desert in the world."
    },
    {
        "title": "Geography of North Africa",
        "text": "North Africa spans the northern portion of the African continent, enveloped largely by the expansive Sahara Desert extending to the Mediterranean Sea."
    },
    {
        "title": "Gobi Desert Asia",
        "text": "The Gobi Desert is a large desert region in East Asia, spanning parts of northern and northeastern China and southern Mongolia."
    }
]

with open("dataset/corpus.json", "w", encoding="utf-8") as f:
    json.dump(CORPUS, f, indent=2)

print(f"Successfully saved dataset/corpus.json with {len(CORPUS)} passages.")
