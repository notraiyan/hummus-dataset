# Related work and novelty

The project extracts, from food reviews, who holds each opinion, which aspect of the recipe it is about, its sentiment, and the occasion or condition it is tied to. It then tests whether keeping the holder and the condition stops a user profile from generalising wrongly, for example by turning a child's dislike into the reviewer's own preference. This note places each part of that pipeline in prior work and states where the project goes beyond it. It is a starting point, not a systematic review: every reference should be checked, and the search extended, before the claims below are used in a report.

## 1. Food recommendation and the HUMMUS data

Food recommenders have mostly modelled what people rate, cook or eat, with growing attention to health (Trattner & Elsweiler, 2017; Elsweiler, Hauptmann & Trattner, 2022). HUMMUS (Bölz et al., 2023), the data set used here, links Food.com recipes, ratings and review texts with nutrition and healthiness scores, so that recommendation and argumentation can draw on more than ratings. Food.com-style interaction data has also been used to personalise recipe generation from a user's history (Majumder et al., 2019).

In this line of work, a user's history, including review text, is treated as evidence of that user's own taste. The project questions exactly that: a review often reports other people's reactions ("too hot for my children") and situational judgements ("too time-consuming for weekdays").

## 2. Context-aware recommendation and context mined from reviews

Context-aware recommender systems add situations such as time, company or occasion to the user-item model (Adomavicius & Tuzhilin, 2011). Because context is rarely recorded, several studies mine it from review text: Hariri, Mobasher, Burke and Zheng (2011) infer contexts from hotel reviews, and Bauman and Tuzhilin (2014; 2022) separate context-rich from generic reviews and extract contextual variables from them, improving recommendation on Yelp data.

These methods recover *which* context a review mentions, but they attach it to the review as a whole, and they assume the reviewer holds every opinion in it. The project instead ties each context to a single opinion and to its holder, so "great for weekend dinners" and "too time-consuming for weekdays" in the same review stay separate preferences.

## 3. Several people behind one profile

Shared-account recommendation addresses accounts used by several people, such as a family account, and tries to serve each of them without knowing who is active (Verstrepen & Goethals, 2015). The underlying problem is the one this project targets: one profile mixing the tastes of different people. Shared-account methods separate users statistically from their interactions; here the review text itself says whose opinion each statement is.

## 4. Aspect-based sentiment and review-based recommendation

Aspect-based sentiment analysis (ABSA) assigns sentiment to aspects within a text (Pontiki et al., 2014); the ABSA model used in Task 4 comes from the PyABSA framework (Yang, Zhang & Li, 2023). Review-based recommenders use such phrase-level opinions to model users and to explain recommendations (Zhang et al., 2014; see the survey by Zhang & Chen, 2020).

These models treat every aspect opinion in a user's review as that user's own, unconditional opinion. The project adds the holder and the condition to each aspect opinion before it enters a profile.

Recent work has large language models write the user profile itself: Ramos et al. (2024) condense a user's earlier reviews into a natural-language profile that the user can read and edit, and recommend from it. Such a profile is written from review text, so it inherits the question this project asks: whose opinions, and under which circumstances, the reviews report. Task 9 measures how often an LLM summarising a member's opinions credits someone else's opinion to the member.

## 5. Opinion holders and structured sentiment

Liu (2012) defines an opinion as a quintuple of entity, aspect, sentiment, holder and time, and introduces it with a camera review whose last sentence is not the author's opinion: "However, my wife thinks it is too heavy for her." Opinion holders (or sources) have otherwise been studied mainly in news and other non-review text: the MPQA corpus annotates opinion sources (Wiebe, Wilson & Cardie, 2005), and early systems identified holders with sequence labelling and extraction patterns (Choi et al., 2005) or semantic role labelling (Kim & Hovy, 2006). Structured sentiment analysis now extracts complete opinion tuples of holder, target, expression and polarity (Barnes et al., 2022).

This is the closest prior work to Task 5. These tuples, however, have no slot for the occasion or condition an opinion is tied to, and they are evaluated as an extraction task, not as input to user profiles. The project's tuple (holder, aspect, sentiment, context, evidence span) keeps the holder and aspect of Liu's quintuple, adds the occasion or condition under which the opinion holds, and uses the tuples to build and test profiles.

## 6. Negation, conditions and exceptions

Negation scope and focus have their own shared task and annotated data (Morante & Blanco, 2012), and conditional sentences have been shown to need dedicated sentiment treatment (Narayanan, Liu & Choudhary, 2009). In preference representation, conditional preferences are formalised, for example, by CP-nets (Boutilier et al., 2004).

Task 6 draws on these ideas but applies them to profile construction: "I enjoy spicy food, except at breakfast" becomes a general preference plus an exception, and a negated cue ("not too spicy") changes the stored sentiment, checked with minimal pairs.

## 7. LLMs for structured opinion extraction

Large language models do well on simple sentiment classification but lag behind smaller task-specific models on tasks that need structured sentiment output (Zhang et al., 2024). The project's comparison in Task 5 is consistent with this: on the hand-written test set, dependency rules reach a tuple accuracy of 0.94 and a 4-billion-parameter instruction model with a fixed output schema 0.68.

## 8. Named entity recognition from dictionaries

Training a neural tagger from dictionary matches (distant supervision) is an established way to build domain NER without annotation (Shang et al., 2018). For food, FoodBase provides recipe text annotated by a rule-based tagger and partly checked by hand (Popovski, Koroušić Seljak & Eftimov, 2019). Task 2 follows the distant-supervision setup; on real reviews its model does not beat the dictionary it learned from, which is why Task 10 also trains a model on the author's checked labels.

## 9. Faithful and correctly attributed explanations

Generated recommendation explanations (e.g. Li, Zhang & Chen, 2021) are known to contain statements their sources do not support (Xie et al., 2023), and attribution of generated text to identified sources can be measured by human judgement (Rashkin et al., 2023). These checks ask whether a statement is supported. Task 9 adds a narrower question that matters for personal explanations: whether an opinion is credited to the right person and kept under its condition.

## 10. Evaluation

Ignoring the global timeline lets models train on interactions from after the test point and makes results incomparable (Ji et al., 2020); Task 8 therefore splits chronologically and builds profiles from earlier reviews only. Ranking the held-out recipe among sampled negatives, as Task 8 does, can change the order of methods compared with ranking all items (Krichene & Rendle, 2020), so its absolute numbers and small differences should be read with care. Topics are compared with standard coherence and diversity measures (Röder, Both & Hinneburg, 2015; Dieng, Ruiz & Blei, 2020) across LDA (Blei, Ng & Jordan, 2003), NMF (Lee & Seung, 1999) and BERTopic (Grootendorst, 2022).

## Novelty

Within the work reviewed here:

- opinion definitions and structured sentiment analysis include the holder but not the occasion or condition of an opinion, and do not feed user profiles;
- context-aware recommendation mines contexts from reviews but assumes the reviewer holds every opinion;
- review-based and LLM-written user profiles treat a user's reviews as the user's own tastes;
- shared-account recommendation addresses profiles that mix people, but without the textual evidence of whose opinion is whose;
- explanation checks test support, not whether an opinion is credited to the right person.

A search for aspect-based sentiment analysis on recipe reviews (Food.com, Allrecipes) found surveys and restaurant-review data sets, but no recipe-specific work. No reviewed work jointly extracts holder, aspect, sentiment and contextual condition from food reviews and tests whether keeping them prevents incorrect generalisation in user profiles and incorrect attribution in explanations. That combination, and the benchmark that measures it, is the project's claimed contribution. A fuller search (for example for "opinion holder" with "recommendation", "situational preference", "multi-user household", and recent LLM work on review-based user modelling) is needed before stating it as established.

## References

- Adomavicius, G., & Tuzhilin, A. (2011). Context-aware recommender systems. In F. Ricci, L. Rokach, B. Shapira & P. B. Kantor (Eds.), *Recommender Systems Handbook* (pp. 217–253). Springer.
- Barnes, J., Oberländer, L., Troiano, E., Kutuzov, A., Buchmann, J., Agerri, R., Øvrelid, L., & Velldal, E. (2022). SemEval 2022 Task 10: Structured sentiment analysis. In *Proceedings of SemEval-2022* (pp. 1280–1295).
- Bauman, K., & Tuzhilin, A. (2014). Discovering contextual information from user reviews for recommendation purposes. In *CBRecSys workshop at RecSys 2014*.
- Bauman, K., & Tuzhilin, A. (2022). Know thy context: Parsing contextual information from user reviews for recommendation purposes. *Information Systems Research*, 33(1), 179–202.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research*, 3, 993–1022.
- Bölz, F., et al. (2023). HUMMUS: A linked, healthiness-aware, user-centered and argument-enabling recipe data set for recommendation. In *Proceedings of RecSys 2023*. https://doi.org/10.1145/3604915.3609491
- Boutilier, C., Brafman, R. I., Domshlak, C., Hoos, H. H., & Poole, D. (2004). CP-nets: A tool for representing and reasoning with conditional ceteris paribus preference statements. *Journal of Artificial Intelligence Research*, 21, 135–191.
- Choi, Y., Cardie, C., Riloff, E., & Patwardhan, S. (2005). Identifying sources of opinions with conditional random fields and extraction patterns. In *Proceedings of HLT/EMNLP 2005*.
- Dieng, A. B., Ruiz, F. J. R., & Blei, D. M. (2020). Topic modeling in embedding spaces. *Transactions of the ACL*, 8, 439–453.
- Elsweiler, D., Hauptmann, H., & Trattner, C. (2022). Food recommender systems. In *Recommender Systems Handbook* (3rd ed.). Springer.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
- Hariri, N., Mobasher, B., Burke, R., & Zheng, Y. (2011). Context-aware recommendation based on review mining. In *Workshop on Intelligent Techniques for Web Personalization and Recommender Systems (ITWP) at IJCAI 2011*.
- Ji, Y., Sun, A., Zhang, J., & Li, C. (2020). A critical study on data leakage in recommender system offline evaluation. arXiv:2010.11060.
- Kim, S.-M., & Hovy, E. (2006). Extracting opinions, opinion holders, and topics expressed in online news media text. In *Proceedings of the Workshop on Sentiment and Subjectivity in Text* (pp. 1–8).
- Krichene, W., & Rendle, S. (2020). On sampled metrics for item recommendation. In *Proceedings of KDD 2020*.
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401, 788–791.
- Li, L., Zhang, Y., & Chen, L. (2021). Personalized transformer for explainable recommendation. In *Proceedings of ACL-IJCNLP 2021* (pp. 4947–4957).
- Liu, B. (2012). *Sentiment Analysis and Opinion Mining*. Morgan & Claypool.
- Majumder, B. P., Li, S., Ni, J., & McAuley, J. (2019). Generating personalized recipes from historical user preferences. In *Proceedings of EMNLP-IJCNLP 2019* (pp. 5976–5982).
- Morante, R., & Blanco, E. (2012). *SEM 2012 shared task: Resolving the scope and focus of negation. In *Proceedings of *SEM 2012* (pp. 265–274).
- Narayanan, R., Liu, B., & Choudhary, A. (2009). Sentiment analysis of conditional sentences. In *Proceedings of EMNLP 2009* (pp. 180–189).
- Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., & Manandhar, S. (2014). SemEval-2014 Task 4: Aspect based sentiment analysis. In *Proceedings of SemEval 2014* (pp. 27–35).
- Popovski, G., Koroušić Seljak, B., & Eftimov, T. (2019). FoodBase corpus: A new resource of annotated food entities. *Database*, 2019, baz121.
- Ramos, J., Rahmani, H. A., Wang, X., Fu, X., & Lipani, A. (2024). Transparent and scrutable recommendations using natural language user profiles. In *Proceedings of ACL 2024*.
- Rashkin, H., et al. (2023). Measuring attribution in natural language generation models. *Computational Linguistics*, 49(4), 777–840.
- Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. In *Proceedings of WSDM 2015* (pp. 399–408).
- Shang, J., Liu, L., Gu, X., Ren, X., Ren, T., & Han, J. (2018). Learning named entity tagger using domain-specific dictionary. In *Proceedings of EMNLP 2018* (pp. 2054–2064).
- Trattner, C., & Elsweiler, D. (2017). Food recommender systems: Important contributions, challenges and future research directions. arXiv:1711.02760.
- Verstrepen, K., & Goethals, B. (2015). Top-N recommendation for shared accounts. In *Proceedings of RecSys 2015* (pp. 59–66).
- Wiebe, J., Wilson, T., & Cardie, C. (2005). Annotating expressions of opinions and emotions in language. *Language Resources and Evaluation*, 39(2–3), 165–210.
- Xie, Z., Singh, S., McAuley, J., & Majumder, B. P. (2023). Factual and informative review generation for explainable recommendation. In *Proceedings of AAAI 2023*, 37(11), 13816–13824.
- Yang, H., Zhang, C., & Li, K. (2023). PyABSA: A modularized framework for reproducible aspect-based sentiment analysis. In *Proceedings of CIKM 2023* (pp. 5117–5122).
- Zhang, W., Deng, Y., Liu, B., Pan, S. J., & Bing, L. (2024). Sentiment analysis in the era of large language models: A reality check. In *Findings of NAACL 2024* (pp. 3881–3906).
- Zhang, Y., & Chen, X. (2020). Explainable recommendation: A survey and new perspectives. *Foundations and Trends in Information Retrieval*, 14(1), 1–101.
- Zhang, Y., Lai, G., Zhang, M., Zhang, Y., Liu, Y., & Ma, S. (2014). Explicit factor models for explainable recommendation based on phrase-level sentiment analysis. In *Proceedings of SIGIR 2014* (pp. 83–92).
