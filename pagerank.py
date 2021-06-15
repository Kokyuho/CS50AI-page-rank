import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    modelPage = page

    # Calculate (1 - damping_factor) probability divided over number of pages
    randomPageProb = (1 - damping_factor) / len(corpus)

    # Create probability distribution dict and populate it accordingly
    probDist = dict()

    # If the model page does not contain links to other pages:
    # print(page)
    # print(corpus[modelPage])
    if len(corpus[modelPage]) == 0:
        linkProb = damping_factor / len(corpus)
        for page in corpus:
            probDist[page] = randomPageProb + linkProb

    # If it does:
    if len(corpus[modelPage]) != 0:
        linkProb = damping_factor / len(corpus[modelPage])
        for page in corpus:
            probDist[page] = randomPageProb
            if page in corpus[modelPage]:
                probDist[page] += linkProb

    return probDist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Get first page
    page = random.choice(list(corpus.keys()))

    # Create pageRank dictionary and populate it with 0s
    pageRank = dict()
    for page in corpus:
        pageRank[page] = 0

    # Sample n times using the transition model on previous page
    for i in range(n):

        # First add page to page rank dictionary count
        pageRank[page] += 1/n

        # Get probability distribution for that page
        probDist = transition_model(corpus, page, damping_factor)

        # Get next page based on probability distribution
        population = []
        weights = []
        for name in probDist:
            population.append(name)
            weights.append(probDist[name])
        page = random.choices(population, weights)[0]

    # FOR DEBUGING
    # sum = 0
    # for x in pageRank:
    #     sum += pageRank[x]
    # print(f"The sum of pageRank values is: {sum}")
    # print(f"The probDist for {page} is: {probDist}")
    # print(f"The sum is: {sum}")

    return pageRank
    

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Constants
    d = damping_factor
    N = len(corpus)

    # Create pageRank dictionary and populate it with 1/N
    pageRank = dict()
    residuals = dict()
    for page in corpus:
        pageRank[page] = 1/N
        residuals[page] = 1

    # While not converged
    convergence = False
    while convergence == False:

        # Run page rank formula for all pages
        pageRankCopy = pageRank.copy()
        for page in corpus:

            # Run formula for page
            pageRank[page] = (1-d)/N + d*sum((pageRank[i]/len(corpus[i]) for i in corpus if page in corpus[i]))

            # Calculate new residual
            residuals[page] = pageRankCopy[page] - pageRank[page]
            
        # Check all residuals to be lower than 0.001 and asign convergence true if so
        convergence = True
        for page in corpus:
            if residuals[page] > 0.001:
                convergence = False
                break

    return pageRank


if __name__ == "__main__":
    main()
