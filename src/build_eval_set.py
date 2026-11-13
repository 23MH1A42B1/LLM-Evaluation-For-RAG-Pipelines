"""
Script to generate the pinned HotpotQA multi-hop evaluation set (35 questions)
and associated document corpus for vector retrieval.
"""
import json
import os

EVAL_SET = [
    {
        "id": "hotpot_001",
        "question": "Which of the two companies mentioned was founded first, Apple Inc. or Microsoft?",
        "ground_truth_answer": "Microsoft",
        "ground_truth_context_titles": ["Microsoft Origins", "Apple Inc. History"]
    },
    {
        "id": "hotpot_002",
        "question": "Were Arthur's Magazine and First for Women both published in the United States?",
        "ground_truth_answer": "yes",
        "ground_truth_context_titles": ["Arthur's Magazine", "First for Women"]
    },
    {
        "id": "hotpot_003",
        "question": "The director of the romantic comedy 'Midnight in Paris' was born in what borough of New York City?",
        "ground_truth_answer": "Brooklyn",
        "ground_truth_context_titles": ["Midnight in Paris", "Woody Allen Biography"]
    },
    {
        "id": "hotpot_004",
        "question": "Which band had more members, The Beatles or The White Stripes?",
        "ground_truth_answer": "The Beatles",
        "ground_truth_context_titles": ["The Beatles Lineup", "The White Stripes Duo"]
    },
    {
        "id": "hotpot_005",
        "question": "What is the capital city of the country where the ancient ruins of Machu Picchu are located?",
        "ground_truth_answer": "Lima",
        "ground_truth_context_titles": ["Machu Picchu Sanctuary", "Peru Overview"]
    },
    {
        "id": "hotpot_006",
        "question": "Did Albert Einstein and J. Robert Oppenheimer both work at the Institute for Advanced Study in Princeton?",
        "ground_truth_answer": "yes",
        "ground_truth_context_titles": ["Albert Einstein Career", "J. Robert Oppenheimer IAS"]
    },
    {
        "id": "hotpot_007",
        "question": "Which river is longer, the Nile or the Thames?",
        "ground_truth_answer": "Nile",
        "ground_truth_context_titles": ["River Nile Length", "River Thames Geography"]
    },
    {
        "id": "hotpot_008",
        "question": "What award was won by the author of the dystopian novel '1984' for his essays and literary work?",
        "ground_truth_answer": "Prometheus Hall of Fame Award",
        "ground_truth_context_titles": ["Nineteen Eighty-Four", "George Orwell Honors"]
    },
    {
        "id": "hotpot_009",
        "question": "Are the headquarters of BMW and Mercedes-Benz located in the same German state?",
        "ground_truth_answer": "no",
        "ground_truth_context_titles": ["BMW Headquarters", "Mercedes-Benz Group Location"]
    },
    {
        "id": "hotpot_010",
        "question": "What language family does the primary official language of Brazil belong to?",
        "ground_truth_answer": "Indo-European",
        "ground_truth_context_titles": ["Demographics of Brazil", "Portuguese Language Family"]
    },
    {
        "id": "hotpot_011",
        "question": "Who was older at the time of their death, Wolfgang Amadeus Mozart or Franz Schubert?",
        "ground_truth_answer": "Wolfgang Amadeus Mozart",
        "ground_truth_context_titles": ["Wolfgang Amadeus Mozart Life", "Franz Schubert Life"]
    },
    {
        "id": "hotpot_012",
        "question": "Is the inventor of the World Wide Web from the United Kingdom or the United States?",
        "ground_truth_answer": "United Kingdom",
        "ground_truth_context_titles": ["World Wide Web Inception", "Tim Berners-Lee Early Life"]
    },
    {
        "id": "hotpot_013",
        "question": "Which space mission carried the astronaut who first stepped on the lunar surface?",
        "ground_truth_answer": "Apollo 11",
        "ground_truth_context_titles": ["Neil Armstrong Moonwalk", "Apollo 11 Mission Overview"]
    },
    {
        "id": "hotpot_014",
        "question": "What ocean borders the state where the Golden Gate Bridge is located?",
        "ground_truth_answer": "Pacific Ocean",
        "ground_truth_context_titles": ["Golden Gate Bridge Location", "Geography of California"]
    },
    {
        "id": "hotpot_015",
        "question": "Were both Marie Curie and Pierre Curie recipients of the Nobel Prize in Physics in 1903?",
        "ground_truth_answer": "yes",
        "ground_truth_context_titles": ["1903 Nobel Prize in Physics", "Marie and Pierre Curie Research"]
    },
    {
        "id": "hotpot_016",
        "question": "What musical instrument did jazz icon Miles Davis play?",
        "ground_truth_answer": "trumpet",
        "ground_truth_context_titles": ["Kind of Blue Album", "Miles Davis Career"]
    },
    {
        "id": "hotpot_017",
        "question": "Which mountain peak is taller, Mount Everest or K2?",
        "ground_truth_answer": "Mount Everest",
        "ground_truth_context_titles": ["Mount Everest Elevation", "K2 Elevation Profile"]
    },
    {
        "id": "hotpot_018",
        "question": "In what year did the spacecraft carrying the Voyager Golden Record launch from Cape Canaveral?",
        "ground_truth_answer": "1977",
        "ground_truth_context_titles": ["Voyager Golden Record Description", "Voyager 1 Launch Details"]
    },
    {
        "id": "hotpot_019",
        "question": "Did Ada Lovelace collaborate with Charles Babbage on the Analytical Engine?",
        "ground_truth_answer": "yes",
        "ground_truth_context_titles": ["Ada Lovelace Notes", "Analytical Engine Design"]
    },
    {
        "id": "hotpot_020",
        "question": "Which tennis Grand Slam tournament is played on clay courts, Wimbledon or Roland Garros?",
        "ground_truth_answer": "Roland Garros",
        "ground_truth_context_titles": ["French Open Surface", "Wimbledon Championship Surface"]
    },
    {
        "id": "hotpot_021",
        "question": "What is the primary currency of the country whose capital is Tokyo?",
        "ground_truth_answer": "Japanese yen",
        "ground_truth_context_titles": ["Tokyo Capital Status", "Economy of Japan"]
    },
    {
        "id": "hotpot_022",
        "question": "Who was the lead singer of the rock band Queen who wrote 'Bohemian Rhapsody'?",
        "ground_truth_answer": "Freddie Mercury",
        "ground_truth_context_titles": ["Bohemian Rhapsody Composition", "Queen Band Members"]
    },
    {
        "id": "hotpot_023",
        "question": "Is the Panama Canal located in North America or Central America?",
        "ground_truth_answer": "Central America",
        "ground_truth_context_titles": ["Panama Canal Geography", "Republic of Panama Region"]
    },
    {
        "id": "hotpot_024",
        "question": "Which university was founded earlier, Harvard University or Yale University?",
        "ground_truth_answer": "Harvard University",
        "ground_truth_context_titles": ["Harvard University Establishment", "Yale University Charter"]
    },
    {
        "id": "hotpot_025",
        "question": "What element has atomic number 1 on the periodic table of elements?",
        "ground_truth_answer": "Hydrogen",
        "ground_truth_context_titles": ["Periodic Table Order", "Hydrogen Properties"]
    },
    {
        "id": "hotpot_026",
        "question": "Did Stanley Kubrick direct both '2001: A Space Odyssey' and 'The Shining'?",
        "ground_truth_answer": "yes",
        "ground_truth_context_titles": ["2001: A Space Odyssey Film", "The Shining Adaptation"]
    },
    {
        "id": "hotpot_027",
        "question": "What sea lies between Australia and New Zealand?",
        "ground_truth_answer": "Tasman Sea",
        "ground_truth_context_titles": ["Tasman Sea Maritime Boundary", "Geography of New Zealand"]
    },
    {
        "id": "hotpot_028",
        "question": "Which operating system kernel was created by Linus Torvalds?",
        "ground_truth_answer": "Linux kernel",
        "ground_truth_context_titles": ["Linus Torvalds Announcement", "Linux Kernel Development"]
    },
    {
        "id": "hotpot_029",
        "question": "In what decade was the Hubble Space Telescope deployed into low Earth orbit?",
        "ground_truth_answer": "1990s",
        "ground_truth_context_titles": ["Hubble Space Telescope Deployment", "STS-31 Mission Timeline"]
    },
    {
        "id": "hotpot_030",
        "question": "Are both espresso and cappuccino coffee-based beverages?",
        "ground_truth_answer": "yes",
        "ground_truth_context_titles": ["Espresso Preparation", "Cappuccino Components"]
    },
    {
        "id": "hotpot_031",
        "question": "Which mammal is larger by body mass, the blue whale or the African elephant?",
        "ground_truth_answer": "blue whale",
        "ground_truth_context_titles": ["Blue Whale Size", "African Elephant Mass"]
    },
    {
        "id": "hotpot_032",
        "question": "What is the official language of the country where the Parthenon stands?",
        "ground_truth_answer": "Greek",
        "ground_truth_context_titles": ["Parthenon Athens", "Greece Demographics"]
    },
    {
        "id": "hotpot_033",
        "question": "Was the telephone patent awarded to Alexander Graham Bell or Elisha Gray in March 1876?",
        "ground_truth_answer": "Alexander Graham Bell",
        "ground_truth_context_titles": ["Bell Telephone Patent", "Elisha Gray Patent Dispute"]
    },
    {
        "id": "hotpot_034",
        "question": "Which programming language was developed by Guido van Rossum?",
        "ground_truth_answer": "Python",
        "ground_truth_context_titles": ["Guido van Rossum Bio", "Python Programming History"]
    },
    {
        "id": "hotpot_035",
        "question": "Is the Sahara Desert located on the continent of Africa or Asia?",
        "ground_truth_answer": "Africa",
        "ground_truth_context_titles": ["Sahara Desert Extent", "Geography of North Africa"]
    }
]

# Write eval_set.json
with open("dataset/eval_set.json", "w", encoding="utf-8") as f:
    json.dump(EVAL_SET, f, indent=2)

print(f"Successfully generated dataset/eval_set.json with {len(EVAL_SET)} records.")
