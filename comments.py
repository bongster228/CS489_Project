"""
Simple module to retrieve a sizeable chunk of a redditor's comment history.
parameters: PRAW reddit instance, reddit username
returns: dictionary of user score by subreddit
"""
import praw

def get_comment_score_per_sub(instance, user):
    redditor = instance.redditor(user)

    temp = {'name': redditor.name}  # dictionary for subreddit scores
    comment_list = []               # list of comment ids to check for duplicate comments
    
    sorts = [redditor.comments.new(limit=None), redditor.comments.top(limit=None), redditor.comments.controversial(limit=None)]
    for sort in sorts:
        for redditor_comment in sort:
            body = redditor_comment.body # Get comment body

            if redditor_comment.id not in comment_list:
                # Update subreddit score if exists
                if redditor_comment.subreddit.display_name in temp.keys():
                    try:
                        temp[redditor_comment.subreddit.display_name] += redditor_comment.score
                    except:
                        temp = {}
                        print("Missing redditor!")

                # Else create subreddit key initialize to comment score
                elif redditor_comment.subreddit.display_name not in temp.keys():
                    try:
                        temp[redditor_comment.subreddit.display_name] = redditor_comment.score
                    except:
                        temp = {}
                        print("Missing redditor!")

                comment_list.append(redditor_comment.id)

    return temp
